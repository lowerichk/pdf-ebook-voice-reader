import json
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import unicodedata

import numpy as np
import onnxruntime as ort

# Silence harmless ONNX Runtime FP16 constant-folding warnings on Apple Silicon
ort.set_default_logger_severity(3)

from kokoro_onnx import Kokoro
from PIL import Image, ImageDraw, ImageTk
import pymupdf as fitz
import sounddevice as sd

# ================= MODEL, BOOKMARK & VOICE CONFIG =================
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / "reader_bookmarks.json"

MODEL_CANDIDATES = [
    SCRIPT_DIR / "kokoro-v1.0.onnx",
    SCRIPT_DIR / "kokoro-v1.0.fp16.onnx",
    SCRIPT_DIR / "kokoro-v1.0.int8.onnx",
]
VOICES_PATH = SCRIPT_DIR / "voices-v1.0.bin"

VOICE_OPTIONS = [
    ("af_heart", "en-us", "af_heart (US Female ★)"),
    ("af_bella", "en-us", "af_bella (US Female)"),
    ("af_sarah", "en-us", "af_sarah (US Female)"),
    ("am_adam", "en-us", "am_adam (US Male ★)"),
    ("am_michael", "en-us", "am_michael (US Male)"),
    ("am_fenrir", "en-us", "am_fenrir (US Male)"),
    ("bf_emma", "en-gb", "bf_emma (UK Female ★)"),
    ("bm_george", "en-gb", "bm_george (UK Male ★)"),
    ("bm_lewis", "en-gb", "bm_lewis (UK Male)"),
]

SPEED_OPTIONS = ["0.8x", "0.9x", "1.0x", "1.1x", "1.25x", "1.4x"]
THEME_NAMES = ["Dark", "Sepia", "Light"]
PREFETCH_QUEUE_SIZE = 5

THEMES = {
    "Dark": {
        "root_bg": "#141417",
        "toolbar_bg": "#222226",
        "opts_bg": "#1b1b1f",
        "canvas_bg": "#121215",
        "panel_bg": "#1b1b1f",
        "text_bg": "#16161a",
        "text_fg": "#e2e2e7",
        "muted_fg": "#a0a0ab",
        "label_fg": "#f0f0f5",
        "entry_bg": "#2c2c32",
        "entry_fg": "#ffffff",
        "status_fg": "#64b5f6",
        "speak_bg": "#1f538d",
        "speak_fg": "#ffffff",
        "sel_bg": "#33353d",
        "sel_fg": "#ffd54f",
        "pdf_speak_fill": (66, 165, 245, 72),
        "pdf_speak_outline": (100, 181, 246, 175),
        "pdf_sel_fill": (255, 213, 79, 42),
        "pdf_sel_outline": (255, 213, 79, 135),
    },
    "Sepia": {
        "root_bg": "#eadfcb",
        "toolbar_bg": "#dfd2ba",
        "opts_bg": "#e6dac4",
        "canvas_bg": "#d8cbb3",
        "panel_bg": "#e6dac4",
        "text_bg": "#f4ecd8",
        "text_fg": "#3b2f25",
        "muted_fg": "#6e5c49",
        "label_fg": "#2f241b",
        "entry_bg": "#fbf6ea",
        "entry_fg": "#2f241b",
        "status_fg": "#8d4925",
        "speak_bg": "#d6b88d",
        "speak_fg": "#1e150d",
        "sel_bg": "#e4d3b5",
        "sel_fg": "#7a3b12",
        "pdf_speak_fill": (235, 155, 35, 76),
        "pdf_speak_outline": (195, 110, 15, 165),
        "pdf_sel_fill": (180, 125, 55, 42),
        "pdf_sel_outline": (155, 95, 30, 130),
    },
    "Light": {
        "root_bg": "#e5e5e8",
        "toolbar_bg": "#f3f3f5",
        "opts_bg": "#ebebee",
        "canvas_bg": "#d4d4d8",
        "panel_bg": "#ebebee",
        "text_bg": "#ffffff",
        "text_fg": "#1f2328",
        "muted_fg": "#57606a",
        "label_fg": "#1f2328",
        "entry_bg": "#ffffff",
        "entry_fg": "#1f2328",
        "status_fg": "#0969da",
        "speak_bg": "#cce5ff",
        "speak_fg": "#0a3069",
        "sel_bg": "#fff8c5",
        "sel_fg": "#7d4e00",
        "pdf_speak_fill": (255, 222, 35, 92),
        "pdf_speak_outline": (230, 175, 0, 185),
        "pdf_sel_fill": (9, 105, 218, 36),
        "pdf_sel_outline": (9, 105, 218, 125),
    },
}

SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F]")
ABBREV_END_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Eq|No|vs|etc|e\.g|i\.e)\.$", re.IGNORECASE)
# ==================================================================


class OfflineKokoroPDFReader:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Offline PDF eBook Reader — Kokoro ONNX")
        self.root.geometry("1340x880")

        # Bring window to front on macOS when launched from Terminal
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

        # Configure custom ttk style for themed TOC Treeview
        self.ttk_style = ttk.Style(self.root)
        try:
            self.ttk_style.theme_use("clam")
        except tk.TclError:
            pass

        # PDF & Canvas state
        self.doc_lock = threading.RLock()
        self.doc: fitz.Document | None = None
        self.current_pdf_path: str | None = None
        self.current_page = 0
        self.zoom = 1.4
        self.base_themed_pil: Image.Image | None = None
        self.tk_image = None
        self.canvas_img_id: int | None = None
        self.img_offset = (10, 10)

        # Table of Contents (TOC) state
        self.toc_entries: list[tuple[str, int]] = []  # (tree_item_id, 0-based page_idx)
        self.toc_item_to_page: dict[str, int] = {}
        self._suppress_toc_select = False

        # Sentence text & exact PDF bounding-box state
        self.page_chunks: list[str] = []
        self.chunk_pdf_rects: list[list[tuple[float, float, float, float]]] = []
        self.chunk_ranges: list[tuple[str, str]] = []
        self.selected_chunk_idx = 0

        # Kokoro local engine state
        self.kokoro: Kokoro | None = None
        self.kokoro_lock = threading.Lock()
        self.model_ready = False

        # Playback & Double-Buffer state
        self.is_reading = False
        self.is_paused = False
        self.session_id = 0
        self.ram_audio_cache: dict[tuple, tuple[np.ndarray, int]] = {}

        # Persistent Bookmark & Preferences state
        self.save_lock = threading.Lock()
        self._save_after_id: str | None = None
        self._restoring_session = False
        self.saved_state: dict = self._read_state_from_disk()

        # Initialize UI variables from saved preferences
        prefs = self.saved_state.get("preferences", {})
        saved_geom = prefs.get("window_geometry", "")
        if isinstance(saved_geom, str) and re.match(r"^\d+x\d+", saved_geom):
            try:
                self.root.geometry(saved_geom)
            except tk.TclError:
                pass

        saved_theme = prefs.get("theme", "Dark")
        self.theme_var = tk.StringVar(value=saved_theme if saved_theme in THEME_NAMES else "Dark")
        self.keep_images_natural_var = tk.BooleanVar(value=bool(prefs.get("preserve_colors", True)))
        self.auto_advance_var = tk.BooleanVar(value=bool(prefs.get("auto_advance", True)))
        self.skip_margins_var = tk.BooleanVar(value=bool(prefs.get("skip_margins", True)))

        # Collapsible TOC & Text panel visibility + widths
        self.toc_panel_visible = bool(prefs.get("show_toc_panel", False))
        saved_toc_w = prefs.get("toc_panel_width", 270)
        self.saved_toc_panel_width = (
            max(190, min(520, int(saved_toc_w))) if isinstance(saved_toc_w, (int, float)) else 270
        )

        self.text_panel_visible = bool(prefs.get("show_text_panel", True))
        saved_panel_w = prefs.get("text_panel_width", 410)
        self.saved_text_panel_width = (
            max(250, min(700, int(saved_panel_w))) if isinstance(saved_panel_w, (int, float)) else 410
        )

        self.voice_labels = [v[2] for v in VOICE_OPTIONS]
        saved_voice = prefs.get("voice", self.voice_labels[0])
        self.voice_var = tk.StringVar(
            value=saved_voice if saved_voice in self.voice_labels else self.voice_labels[0]
        )

        saved_speed = prefs.get("speed", "1.0x")
        self.speed_var = tk.StringVar(value=saved_speed if saved_speed in SPEED_OPTIONS else "1.0x")

        saved_default_zoom = prefs.get("default_zoom", 1.4)
        if isinstance(saved_default_zoom, (int, float)):
            self.zoom = max(0.6, min(3.0, round(float(saved_default_zoom), 2)))

        self._build_ui()
        self._apply_ui_theme()
        self._bind_shortcuts()

        # Catch both window close button and macOS Cmd+Q Quit event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if sys.platform == "darwin":
            try:
                self.root.createcommand("tk::mac::Quit", self._on_close)
            except Exception:
                pass

        # Restore last opened PDF, page, sentence & zoom once Tkinter finishes layout
        self.root.after(80, self._restore_last_book_on_startup)

        # Load Kokoro ONNX model in background thread
        threading.Thread(target=self._init_kokoro_engine, daemon=True).start()

    # ================= BOOKMARK & RECENT BOOKS ENGINE =================

    @staticmethod
    def _read_state_from_disk() -> dict:
        if not STATE_FILE.exists():
            return {"preferences": {}, "books": {}, "last_pdf": None}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("preferences", {})
                data.setdefault("books", {})
                data.setdefault("last_pdf", None)
                return data
        except Exception:
            pass
        return {"preferences": {}, "books": {}, "last_pdf": None}

    def schedule_save_state(self, delay_ms: int = 350):
        if self._restoring_session:
            return
        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except Exception:
                pass
        self._save_after_id = self.root.after(delay_ms, self.save_state_now)

    def save_state_now(self):
        if self._restoring_session:
            return
        self._save_after_id = None

        if self.toc_panel_visible and self.toc_frame.winfo_width() > 160:
            self.saved_toc_panel_width = self.toc_frame.winfo_width()
        if self.text_panel_visible and self.text_frame.winfo_width() > 200:
            self.saved_text_panel_width = self.text_frame.winfo_width()

        with self.save_lock:
            prefs = {
                "voice": self.voice_var.get(),
                "speed": self.speed_var.get(),
                "theme": self.theme_var.get(),
                "preserve_colors": bool(self.keep_images_natural_var.get()),
                "auto_advance": bool(self.auto_advance_var.get()),
                "skip_margins": bool(self.skip_margins_var.get()),
                "show_toc_panel": bool(self.toc_panel_visible),
                "toc_panel_width": int(self.saved_toc_panel_width),
                "show_text_panel": bool(self.text_panel_visible),
                "text_panel_width": int(self.saved_text_panel_width),
                "default_zoom": round(self.zoom, 2),
                "window_geometry": self.root.geometry(),
            }
            self.saved_state["preferences"] = prefs

            if self.current_pdf_path and self.doc is not None:
                norm_path = str(Path(self.current_pdf_path).resolve())
                self.saved_state["last_pdf"] = norm_path
                books = self.saved_state.setdefault("books", {})
                books[norm_path] = {
                    "page": int(self.current_page),
                    "sentence": int(self.selected_chunk_idx),
                    "zoom": round(self.zoom, 2),
                    "total_pages": int(self.get_total_pages()),
                    "updated_at": int(time.time()),
                }

                if len(books) > 50:
                    oldest_k = min(books.keys(), key=lambda k: books[k].get("updated_at", 0))
                    if oldest_k != norm_path:
                        books.pop(oldest_k, None)

            try:
                tmp_fd, tmp_path = tempfile.mkstemp(prefix=".bookmarks_", suffix=".json", dir=str(SCRIPT_DIR))
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(self.saved_state, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, STATE_FILE)
            except Exception:
                pass

    def _restore_last_book_on_startup(self):
        last_pdf = self.saved_state.get("last_pdf")
        if last_pdf and os.path.isfile(last_pdf):
            self._load_pdf_file(last_pdf, is_startup_restore=True)

    def _populate_recent_books_menu(self):
        """Rebuilds the Recent Books dropdown dynamically whenever clicked."""
        self.recent_menu.delete(0, tk.END)
        books_dict: dict = self.saved_state.get("books", {})

        # Sort books by most recently read first
        sorted_items = sorted(
            books_dict.items(),
            key=lambda kv: kv[1].get("updated_at", 0) if isinstance(kv[1], dict) else 0,
            reverse=True,
        )

        added = 0
        missing_count = 0
        cur_norm = str(Path(self.current_pdf_path).resolve()) if self.current_pdf_path else ""

        for pdf_path, info in sorted_items:
            if not os.path.isfile(pdf_path):
                missing_count += 1
                continue
            if added >= 18:
                break

            page_num = int(info.get("page", 0)) + 1
            total_p = int(info.get("total_pages", 0))
            name = os.path.basename(pdf_path)
            if len(name) > 44:
                name = name[:41] + "..."

            progress_str = f"Page {page_num}/{total_p}" if total_p > 0 else f"Page {page_num}"
            prefix = "✓ " if pdf_path == cur_norm else "    "
            label = f"{prefix}{name}   ({progress_str})"

            self.recent_menu.add_command(
                label=label,
                command=lambda p=pdf_path: self._switch_to_recent_book(p),
            )
            added += 1

        if added == 0:
            self.recent_menu.add_command(label="No recent books saved yet", state=tk.DISABLED)
        else:
            self.recent_menu.add_separator()
            if missing_count > 0:
                self.recent_menu.add_command(
                    label=f"Prune {missing_count} moved/missing file(s)",
                    command=self._prune_missing_books,
                )
            self.recent_menu.add_command(
                label="Clear other recent books",
                command=self._clear_other_recent_books,
            )

    def _switch_to_recent_book(self, pdf_path: str):
        if self.current_pdf_path and str(Path(self.current_pdf_path).resolve()) == str(Path(pdf_path).resolve()):
            return
        if not os.path.isfile(pdf_path):
            messagebox.showwarning("File Not Found", f"Could not find:\n{pdf_path}")
            self._prune_missing_books()
            return
        self.save_state_now()
        self._load_pdf_file(pdf_path, is_startup_restore=False)

    def _prune_missing_books(self):
        books = self.saved_state.get("books", {})
        to_remove = [p for p in books if not os.path.isfile(p)]
        for p in to_remove:
            books.pop(p, None)
        self.save_state_now()

    def _clear_other_recent_books(self):
        books = self.saved_state.get("books", {})
        cur_norm = str(Path(self.current_pdf_path).resolve()) if self.current_pdf_path else None
        kept = {cur_norm: books[cur_norm]} if (cur_norm and cur_norm in books) else {}
        self.saved_state["books"] = kept
        self.save_state_now()

    # ================= KOKORO ENGINE INIT =================

    def _init_kokoro_engine(self):
        model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
        if not model_path or not VOICES_PATH.exists():
            msg = (
                f"Missing local model files in:\n{SCRIPT_DIR}\n\n"
                "Please ensure 'kokoro-v1.0.fp16.onnx' (or 'kokoro-v1.0.onnx') and 'voices-v1.0.bin' are in the folder."
            )
            self.root.after(0, lambda: self.status_lbl.config(text="Model files missing!", fg="#ef5350"))
            self.root.after(100, lambda: messagebox.showwarning("Kokoro Model Missing", msg))
            return

        try:
            self.root.after(0, lambda: self.buffer_lbl.config(text=f"Loading {model_path.name}...", fg="#ffb74d"))
            engine = Kokoro(str(model_path), str(VOICES_PATH))
            engine.create("Ready.", voice="af_heart", speed=1.0, lang="en-us")
            self.kokoro = engine
            self.model_ready = True

            def _on_ready():
                self.buffer_lbl.config(text=f"Engine: {model_path.name} (Offline ✓)", fg="#66bb6a")
                if not self.doc:
                    self.status_lbl.config(
                        text="Offline Kokoro Ready ✓ Open a PDF to begin.",
                        fg=THEMES[self.theme_var.get()]["status_fg"],
                    )

            self.root.after(0, _on_ready)
        except Exception as exc:
            self.root.after(0, lambda e=exc: self.status_lbl.config(text=f"Model Load Error: {e}", fg="#ef5350"))

    # ================= UI CONSTRUCTION & THEMING =================

    def _build_ui(self):
        self.toolbar = tk.Frame(self.root, pady=6, padx=6)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # File, Recent Books & TOC Sidebar buttons
        ttk.Button(self.toolbar, text="Open", command=self.open_pdf, width=6).pack(side=tk.LEFT, padx=2)

        self.recent_btn = ttk.Menubutton(self.toolbar, text="Recent ▾", width=8)
        self.recent_menu = tk.Menu(self.recent_btn, tearoff=0, postcommand=self._populate_recent_books_menu)
        self.recent_btn["menu"] = self.recent_menu
        self.recent_btn.pack(side=tk.LEFT, padx=2)

        toc_btn_txt = "Hide TOC (C)" if self.toc_panel_visible else "TOC (C)"
        self.toggle_toc_btn = ttk.Button(self.toolbar, text=toc_btn_txt, command=self.toggle_toc_panel, width=10)
        self.toggle_toc_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # Page Navigation & Zoom
        ttk.Button(self.toolbar, text="◀", command=self.prev_page, width=3).pack(side=tk.LEFT, padx=1)

        self.page_entry = tk.Entry(self.toolbar, width=5, justify="center", relief=tk.FLAT)
        self.page_entry.pack(side=tk.LEFT, padx=2)
        self.page_entry.bind("<Return>", self.jump_to_page)

        self.total_pages_lbl = tk.Label(self.toolbar, text="/ 0")
        self.total_pages_lbl.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.toolbar, text="▶", command=self.next_page, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(self.toolbar, text="Zoom -", command=lambda: self.change_zoom(-0.2), width=6).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(self.toolbar, text="Zoom +", command=lambda: self.change_zoom(0.2), width=6).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # --- Audio Controls ---
        self.play_btn = ttk.Button(self.toolbar, text="▶ Play", command=self.start_reading, width=7)
        self.play_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(self.toolbar, text="⏸ Pause", command=self.toggle_pause, state=tk.DISABLED, width=8)
        self.pause_btn.pack(side=tk.LEFT, padx=1)

        self.stop_btn = ttk.Button(self.toolbar, text="⏹ Stop", command=self.stop_reading, state=tk.DISABLED, width=7)
        self.stop_btn.pack(side=tk.LEFT, padx=1)

        self.restart_btn = ttk.Button(self.toolbar, text="⏮", command=self.restart_page_audio, width=3)
        self.restart_btn.pack(side=tk.LEFT, padx=1)

        ttk.Button(self.toolbar, text="⏪", command=lambda: self.step_sentence(-1), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(self.toolbar, text="⏩", command=lambda: self.step_sentence(1), width=3).pack(side=tk.LEFT, padx=1)

        # Voice, Speed, Theme & Text-Panel Toggle
        self.voice_lbl = tk.Label(self.toolbar, text="Voice:")
        self.voice_lbl.pack(side=tk.LEFT, padx=(5, 2))
        voice_cb = ttk.Combobox(
            self.toolbar, textvariable=self.voice_var, values=self.voice_labels, state="readonly", width=17
        )
        voice_cb.pack(side=tk.LEFT, padx=2)
        voice_cb.bind("<<ComboboxSelected>>", self._on_voice_or_speed_changed)

        self.speed_lbl = tk.Label(self.toolbar, text="Speed:")
        self.speed_lbl.pack(side=tk.LEFT, padx=(4, 2))
        speed_cb = ttk.Combobox(
            self.toolbar, textvariable=self.speed_var, values=SPEED_OPTIONS, state="readonly", width=5
        )
        speed_cb.pack(side=tk.LEFT, padx=2)
        speed_cb.bind("<<ComboboxSelected>>", self._on_voice_or_speed_changed)

        self.theme_lbl = tk.Label(self.toolbar, text="Theme:")
        self.theme_lbl.pack(side=tk.LEFT, padx=(4, 2))
        theme_cb = ttk.Combobox(
            self.toolbar, textvariable=self.theme_var, values=THEME_NAMES, state="readonly", width=6
        )
        theme_cb.pack(side=tk.LEFT, padx=2)
        theme_cb.bind("<<ComboboxSelected>>", self._on_theme_changed)

        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        btn_txt = "Hide Text (T)" if self.text_panel_visible else "Show Text (T)"
        self.toggle_panel_btn = ttk.Button(
            self.toolbar, text=btn_txt, command=self.toggle_text_panel, width=11
        )
        self.toggle_panel_btn.pack(side=tk.LEFT, padx=2)

        # --- Options & Status Bar ---
        self.opts_bar = tk.Frame(self.root, pady=4, padx=10)
        self.opts_bar.pack(side=tk.TOP, fill=tk.X)

        self.auto_adv_cb = tk.Checkbutton(
            self.opts_bar,
            text="Auto-advance pages",
            variable=self.auto_advance_var,
            command=self.schedule_save_state,
        )
        self.auto_adv_cb.pack(side=tk.LEFT, padx=6)

        self.skip_margins_cb = tk.Checkbutton(
            self.opts_bar,
            text="Skip headers & footers (6%)",
            variable=self.skip_margins_var,
            command=self._on_toggle_margins,
        )
        self.skip_margins_cb.pack(side=tk.LEFT, padx=8)

        self.smart_hue_cb = tk.Checkbutton(
            self.opts_bar,
            text="Preserve diagram colors ('C' TOC • 'D' theme • 'T' text panel)",
            variable=self.keep_images_natural_var,
            command=self._on_toggle_preserve_colors,
        )
        self.smart_hue_cb.pack(side=tk.LEFT, padx=8)

        self.buffer_lbl = tk.Label(self.opts_bar, text="Engine: Initializing...")
        self.buffer_lbl.pack(side=tk.RIGHT, padx=(12, 4))

        self.status_lbl = tk.Label(self.opts_bar, text="Ready.")
        self.status_lbl.pack(side=tk.RIGHT, padx=8)

        # --- 3-Pane Main Split View: [TOC Sidebar] | [PDF Canvas] | [Sentence Text Panel] ---
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # 1. Left Collapsible TOC Sidebar
        self.toc_frame = tk.Frame(self.paned)
        self.toc_header_lbl = tk.Label(
            self.toc_frame,
            text="Table of Contents",
            font=("Helvetica Neue", 12, "bold") if sys.platform == "darwin" else ("Segoe UI", 10, "bold"),
            pady=6,
            padx=8,
            anchor="w",
        )
        self.toc_header_lbl.pack(side=tk.TOP, fill=tk.X)

        toc_scroll = tk.Scrollbar(self.toc_frame, orient=tk.VERTICAL)
        self.toc_tree = ttk.Treeview(
            self.toc_frame,
            columns=("page",),
            show="tree headings",
            selectmode="browse",
            style="TOC.Treeview",
            yscrollcommand=toc_scroll.set,
        )
        self.toc_tree.heading("#0", text="Chapter / Section", anchor="w")
        self.toc_tree.heading("page", text="Pg", anchor="e")
        self.toc_tree.column("#0", width=195, minwidth=120, stretch=True)
        self.toc_tree.column("page", width=46, minwidth=38, stretch=False, anchor="e")

        toc_scroll.config(command=self.toc_tree.yview)
        toc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.toc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.toc_tree.bind("<<TreeviewSelect>>", self._on_toc_item_selected)

        # 2. Center Panel: PDF Canvas
        self.canvas_frame = tk.Frame(self.paned)
        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.canvas = tk.Canvas(
            self.canvas_frame,
            highlightthickness=0,
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 3. Right Collapsible Panel: Clickable Sentences
        self.text_frame = tk.Frame(self.paned)
        self.hint_lbl = tk.Label(
            self.text_frame,
            text="Click sentence (here or on PDF) to select • Double-click to play",
            pady=5,
        )
        self.hint_lbl.pack(side=tk.TOP, fill=tk.X)

        text_scroll = tk.Scrollbar(self.text_frame, orient=tk.VERTICAL)
        self.text_box = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            font=("Helvetica Neue", 13) if sys.platform == "darwin" else ("Segoe UI", 11),
            padx=12,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT,
            state=tk.DISABLED,
            yscrollcommand=text_scroll.set,
        )
        text_scroll.config(command=self.text_box.yview)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add visible panes in strict Left -> Center -> Right order
        if self.toc_panel_visible:
            self.paned.add(self.toc_frame, width=self.saved_toc_panel_width, stretch="never")
        self.paned.add(self.canvas_frame, stretch="always")
        if self.text_panel_visible:
            self.paned.add(self.text_frame, width=self.saved_text_panel_width, stretch="never")

    # ================= COLLAPSIBLE TOC & TEXT PANELS =================

    def toggle_toc_panel(self, _event=None):
        """Shows or hides the left-hand Table of Contents chapter drawer ('C' shortcut)."""
        if self.root.focus_get() == self.page_entry:
            return

        if self.toc_panel_visible:
            cur_w = self.toc_frame.winfo_width()
            if cur_w > 160:
                self.saved_toc_panel_width = cur_w
            self.paned.forget(self.toc_frame)
            self.toc_panel_visible = False
            self.toggle_toc_btn.config(text="TOC (C)")
        else:
            total_w = max(900, self.paned.winfo_width())
            toc_w = min(self.saved_toc_panel_width, total_w // 3)
            # Insert directly before canvas_frame so TOC always sits on the far left!
            self.paned.add(self.toc_frame, before=self.canvas_frame, width=toc_w, stretch="never")
            self.toc_panel_visible = True
            self.toggle_toc_btn.config(text="Hide TOC (C)")
            self.root.after(20, self._sync_toc_highlight_to_page)

        self.root.after(25, self._recenter_pdf_on_canvas)
        self.schedule_save_state()

    def toggle_text_panel(self, _event=None):
        """Hides or restores the right-hand text panel ('T' shortcut)."""
        if self.root.focus_get() == self.page_entry:
            return

        if self.text_panel_visible:
            current_w = self.text_frame.winfo_width()
            if current_w > 200:
                self.saved_text_panel_width = current_w
            self.paned.forget(self.text_frame)
            self.text_panel_visible = False
            self.toggle_panel_btn.config(text="Show Text (T)")
        else:
            total_w = max(800, self.paned.winfo_width())
            panel_w = min(self.saved_text_panel_width, total_w // 2)
            self.paned.add(self.text_frame, width=panel_w, stretch="never")
            self.text_panel_visible = True
            self.toggle_panel_btn.config(text="Hide Text (T)")
            if self.page_chunks and 0 <= self.selected_chunk_idx < len(self.chunk_ranges):
                self.root.after(20, lambda: self.text_box.see(self.chunk_ranges[self.selected_chunk_idx][0]))

        self.root.after(25, self._recenter_pdf_on_canvas)
        self.schedule_save_state()

    def _populate_toc_tree(self):
        """Extracts nested chapter bookmarks from the open PDF via PyMuPDF get_toc()."""
        for item in self.toc_tree.get_children():
            self.toc_tree.delete(item)
        self.toc_entries.clear()
        self.toc_item_to_page.clear()

        if not self.doc:
            return

        with self.doc_lock:
            raw_toc = self.doc.get_toc(simple=True)
            total_p = len(self.doc)

        if not raw_toc:
            self.toc_header_lbl.config(text="Table of Contents (None)")
            self.toc_tree.insert("", "end", text="No embedded chapters in PDF", values=("",))
            return

        self.toc_header_lbl.config(text=f"Chapters ({len(raw_toc)})")

        # Stack of (level, tree_item_id) to preserve nested hierarchy cleanly
        level_stack: list[tuple[int, str]] = []

        for entry in raw_toc:
            if len(entry) < 3:
                continue
            lvl, title, page_1based = int(entry[0]), str(entry[1]).strip(), int(entry[2])
            if not title:
                title = "Untitled Section"
            page_idx = max(0, min(page_1based - 1, total_p - 1))

            while level_stack and level_stack[-1][0] >= lvl:
                level_stack.pop()

            parent_id = level_stack[-1][1] if level_stack else ""
            # Auto-expand top-level chapters so sub-sections are easy to browse
            item_id = self.toc_tree.insert(
                parent_id,
                "end",
                text=title,
                values=(str(page_idx + 1),),
                open=(lvl <= 1),
            )
            level_stack.append((lvl, item_id))
            self.toc_entries.append((item_id, page_idx))
            self.toc_item_to_page[item_id] = page_idx

        self._sync_toc_highlight_to_page()

    def _sync_toc_highlight_to_page(self):
        """Highlights the active chapter in the TOC tree corresponding to current_page."""
        if not self.toc_entries:
            return

        active_item = None
        for item_id, p_idx in self.toc_entries:
            if p_idx <= self.current_page:
                active_item = item_id
            else:
                break

        if active_item is None:
            active_item = self.toc_entries[0][0]

        current_sel = self.toc_tree.selection()
        if current_sel and current_sel[0] == active_item:
            return

        self._suppress_toc_select = True
        try:
            self.toc_tree.selection_set(active_item)
            if self.toc_panel_visible:
                self.toc_tree.see(active_item)
        finally:
            self.root.after(30, lambda: setattr(self, "_suppress_toc_select", False))

    def _on_toc_item_selected(self, _event=None):
        if self._suppress_toc_select or not self.doc:
            return
        sel = self.toc_tree.selection()
        if not sel:
            return
        target_page = self.toc_item_to_page.get(sel[0])
        if target_page is None or target_page == self.current_page:
            return

        was_reading = self.is_reading
        self.stop_reading(preserve_position=False)
        self.current_page = target_page
        self.render_current_page(reset_chunk_idx=True)
        self.schedule_save_state()
        if was_reading:
            self.start_reading(from_chunk=0)

    def _recenter_pdf_on_canvas(self, _event=None):
        if self.base_themed_pil is None or self.canvas_img_id is None:
            return
        pix_w, pix_h = self.base_themed_pil.size
        canvas_w = max(self.canvas.winfo_width(), pix_w + 20)
        img_x = max(10, (canvas_w - pix_w) // 2)
        img_y = 10
        self.img_offset = (img_x, img_y)
        self.canvas.coords(self.canvas_img_id, img_x, img_y)
        self.canvas.config(scrollregion=(0, 0, canvas_w, pix_h + 20))

    # ================= THEMING =================

    def _apply_ui_theme(self):
        t = THEMES[self.theme_var.get()]

        self.root.configure(bg=t["root_bg"])
        self.toolbar.configure(bg=t["toolbar_bg"])
        self.opts_bar.configure(bg=t["opts_bg"])
        self.paned.configure(bg=t["root_bg"])
        self.toc_frame.configure(bg=t["panel_bg"])
        self.toc_header_lbl.configure(bg=t["panel_bg"], fg=t["label_fg"])
        self.canvas_frame.configure(bg=t["canvas_bg"])
        self.canvas.configure(bg=t["canvas_bg"])
        self.text_frame.configure(bg=t["panel_bg"])

        # Style the TOC Treeview & Recent Books popup menu to match Dark / Sepia / Light
        self.ttk_style.configure(
            "TOC.Treeview",
            background=t["text_bg"],
            fieldbackground=t["text_bg"],
            foreground=t["text_fg"],
            rowheight=26,
            borderwidth=0,
            font=("Helvetica Neue", 12) if sys.platform == "darwin" else ("Segoe UI", 10),
        )
        self.ttk_style.configure(
            "TOC.Treeview.Heading",
            background=t["toolbar_bg"],
            foreground=t["muted_fg"],
            relief="flat",
        )
        self.ttk_style.map(
            "TOC.Treeview",
            background=[("selected", t["speak_bg"])],
            foreground=[("selected", t["speak_fg"])],
        )
        self.recent_menu.configure(
            bg=t["panel_bg"],
            fg=t["text_fg"],
            activebackground=t["speak_bg"],
            activeforeground=t["speak_fg"],
        )

        self.page_entry.configure(bg=t["entry_bg"], fg=t["entry_fg"], insertbackground=t["entry_fg"])
        for lbl in (self.total_pages_lbl, self.voice_lbl, self.speed_lbl, self.theme_lbl):
            lbl.configure(bg=t["toolbar_bg"], fg=t["label_fg"])

        for cb in (self.auto_adv_cb, self.skip_margins_cb, self.smart_hue_cb):
            cb.configure(
                bg=t["opts_bg"],
                fg=t["label_fg"],
                selectcolor=t["opts_bg"],
                activebackground=t["opts_bg"],
                activeforeground=t["label_fg"],
            )

        self.status_lbl.configure(bg=t["opts_bg"], fg=t["status_fg"])
        self.buffer_lbl.configure(bg=t["opts_bg"])
        self.hint_lbl.configure(bg=t["panel_bg"], fg=t["muted_fg"])

        self.text_box.configure(bg=t["text_bg"], fg=t["text_fg"], insertbackground=t["text_fg"])
        self.text_box.tag_configure("speaking", background=t["speak_bg"], foreground=t["speak_fg"])
        self.text_box.tag_configure("selected_start", background=t["sel_bg"], foreground=t["sel_fg"], underline=True)

    def _apply_pdf_page_theme(self, img: Image.Image) -> Image.Image:
        mode = self.theme_var.get()
        if mode == "Light":
            return img.convert("RGBA")

        arr = np.asarray(img, dtype=np.float32)

        if mode == "Sepia":
            sepia_paper = np.array([244.0, 236.0, 216.0], dtype=np.float32) / 255.0
            tinted = np.clip(arr * sepia_paper, 0, 255).astype(np.uint8)
            return Image.fromarray(tinted, mode="RGB").convert("RGBA")

        # === DARK MODE ===
        if self.keep_images_natural_var.get():
            norm = arr / 255.0
            cmax = norm.max(axis=2)
            cmin = norm.min(axis=2)
            chroma = cmax - cmin

            lum = 0.299 * norm[:, :, 0] + 0.587 * norm[:, :, 1] + 0.114 * norm[:, :, 2]
            target_lum = 0.10 + (1.0 - lum) * 0.81

            delta = (target_lum - lum)[:, :, None]
            sat_damp = np.where(chroma > 0.25, 0.85, 1.0)[:, :, None]
            mean_l = lum[:, :, None]
            recolored = (mean_l + (norm - mean_l) * sat_damp) + delta

            recolored[:, :, 2] += 0.012 * (1.0 - chroma)
            out = np.clip(recolored * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(out, mode="RGB").convert("RGBA")
        else:
            inverted = 255.0 - arr
            softened = 24.0 + inverted * (212.0 / 255.0)
            return Image.fromarray(softened.astype(np.uint8), mode="RGB").convert("RGBA")

    # ================= TRANSLUCENT PDF SENTENCE OVERLAY =================

    @staticmethod
    def _merge_adjacent_line_rects(
        rects: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        if not rects:
            return []
        sorted_r = sorted(rects, key=lambda r: (round(r[1], 1), r[0]))
        merged: list[list[float]] = [list(sorted_r[0])]

        for x0, y0, x1, y1 in sorted_r[1:]:
            prev = merged[-1]
            prev_h = max(1.0, prev[3] - prev[1])
            if abs(((y0 + y1) * 0.5) - ((prev[1] + prev[3]) * 0.5)) < prev_h * 0.55:
                prev[0] = min(prev[0], x0)
                prev[1] = min(prev[1], y0)
                prev[2] = max(prev[2], x1)
                prev[3] = max(prev[3], y1)
            else:
                merged.append([x0, y0, x1, y1])

        return [(r[0], r[1], r[2], r[3]) for r in merged]

    def _draw_pdf_sentence_highlight(
        self, chunk_idx: int | None, style: str = "speaking", auto_scroll: bool = True
    ):
        if self.base_themed_pil is None or self.canvas_img_id is None:
            return

        if chunk_idx is None or not (0 <= chunk_idx < len(self.chunk_pdf_rects)):
            self.tk_image = ImageTk.PhotoImage(self.base_themed_pil)
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_image)
            return

        rects = self.chunk_pdf_rects[chunk_idx]
        if not rects:
            self.tk_image = ImageTk.PhotoImage(self.base_themed_pil)
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_image)
            return

        t = THEMES[self.theme_var.get()]
        if style == "speaking":
            fill_rgba = t["pdf_speak_fill"]
            outline_rgba = t["pdf_speak_outline"]
        else:
            fill_rgba = t["pdf_sel_fill"]
            outline_rgba = t["pdf_sel_outline"]

        overlay = Image.new("RGBA", self.base_themed_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = self.base_themed_pil.size
        pad_x = max(2, int(3 * self.zoom))
        pad_y = max(1, int(2 * self.zoom))
        radius = max(3, int(4 * self.zoom))

        min_px_y = h
        max_px_y = 0

        for x0, y0, x1, y1 in rects:
            px0 = max(0, int(x0 * self.zoom) - pad_x)
            py0 = max(0, int(y0 * self.zoom) - pad_y)
            px1 = min(w - 1, int(x1 * self.zoom) + pad_x)
            py1 = min(h - 1, int(y1 * self.zoom) + pad_y)
            if px1 <= px0 or py1 <= py0:
                continue

            min_px_y = min(min_px_y, py0)
            max_px_y = max(max_px_y, py1)

            draw.rounded_rectangle(
                [px0, py0, px1, py1],
                radius=radius,
                fill=fill_rgba,
                outline=outline_rgba,
                width=1,
            )

        composited = Image.alpha_composite(self.base_themed_pil, overlay)
        self.tk_image = ImageTk.PhotoImage(composited)
        self.canvas.itemconfig(self.canvas_img_id, image=self.tk_image)

        if auto_scroll and max_px_y > min_px_y:
            total_scroll_h = max(1, h + 20)
            canvas_view_h = max(1, self.canvas.winfo_height())
            top_frac, bot_frac = self.canvas.yview()

            img_y = self.img_offset[1]
            box_top_frac = (img_y + min_px_y) / total_scroll_h
            box_bot_frac = (img_y + max_px_y) / total_scroll_h
            margin_frac = 0.08 * (canvas_view_h / total_scroll_h)

            if box_bot_frac > bot_frac - margin_frac or box_top_frac < top_frac + margin_frac:
                target_top = max(0.0, box_top_frac - 0.22 * (canvas_view_h / total_scroll_h))
                self.canvas.yview_moveto(target_top)

    def _on_toggle_preserve_colors(self):
        if self.doc:
            self.render_current_page(reset_chunk_idx=False)
            if self.is_reading:
                self._highlight_active_chunk(self.selected_chunk_idx)
        self.schedule_save_state()

    def _on_theme_changed(self, _event=None):
        self._apply_ui_theme()
        if self.doc:
            self.render_current_page(reset_chunk_idx=False)
            if self.is_reading:
                self._highlight_active_chunk(self.selected_chunk_idx)
        self.schedule_save_state()

    def cycle_theme(self, _event=None):
        if self.root.focus_get() == self.page_entry:
            return
        cur = self.theme_var.get()
        nxt = THEME_NAMES[(THEME_NAMES.index(cur) + 1) % len(THEME_NAMES)]
        self.theme_var.set(nxt)
        self._on_theme_changed()

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda e: self.prev_page() if self.root.focus_get() != self.page_entry else None)
        self.root.bind("<Right>", lambda e: self.next_page() if self.root.focus_get() != self.page_entry else None)
        self.root.bind("<space>", self._on_spacebar)
        self.root.bind("c", self.toggle_toc_panel)
        self.root.bind("C", self.toggle_toc_panel)
        self.root.bind("d", self.cycle_theme)
        self.root.bind("D", self.cycle_theme)
        self.root.bind("t", self.toggle_text_panel)
        self.root.bind("T", self.toggle_text_panel)

        self.canvas.bind("<Configure>", self._recenter_pdf_on_canvas)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

        self.text_box.bind("<Button-1>", self._on_textbox_click)
        self.text_box.bind("<Double-Button-1>", self._on_textbox_double_click)
        self.canvas.bind("<Button-1>", self._on_pdf_canvas_single_click)
        self.canvas.bind("<Double-Button-1>", self._on_pdf_canvas_double_click)

    def _get_voice_and_lang(self) -> tuple[str, str]:
        selected_lbl = self.voice_var.get()
        for vid, lang, lbl in VOICE_OPTIONS:
            if lbl == selected_lbl:
                return vid, lang
        return "af_heart", "en-us"

    def _get_speed_float(self) -> float:
        try:
            return float(self.speed_var.get().replace("x", ""))
        except ValueError:
            return 1.0

    def _on_spacebar(self, _event):
        if self.root.focus_get() == self.page_entry:
            return
        if self.is_reading:
            self.toggle_pause()
        else:
            self.start_reading()

    def _on_mousewheel(self, event):
        if sys.platform == "darwin":
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_voice_or_speed_changed(self, _event=None):
        self.schedule_save_state()
        if self.doc and self.is_reading:
            self.start_reading(from_chunk=self.selected_chunk_idx)

    def _on_toggle_margins(self):
        self.schedule_save_state()
        if not self.doc:
            return
        was_reading = self.is_reading
        self.stop_reading(preserve_position=False)
        self.render_current_page(reset_chunk_idx=True)
        if was_reading:
            self.start_reading(from_chunk=0)

    # ================= CLICK-TO-START (BOTH PANEL & PDF CANVAS) =================

    def _chunk_idx_from_textbox_xy(self, x: int, y: int) -> int | None:
        if not self.page_chunks:
            return None
        click_pos = self.text_box.index(f"@{x},{y}")

        for tag in self.text_box.tag_names(click_pos):
            if tag.startswith("chunk_"):
                return int(tag.split("_")[1])

        for idx, (_start, end) in enumerate(self.chunk_ranges):
            if self.text_box.compare(click_pos, "<=", end):
                return idx
        return len(self.page_chunks) - 1

    def _chunk_idx_from_pdf_xy(self, event_x: int, event_y: int) -> int | None:
        if not self.doc or not self.chunk_pdf_rects:
            return None

        cx = self.canvas.canvasx(event_x)
        cy = self.canvas.canvasy(event_y)
        img_x, img_y = self.img_offset
        pdf_x = (cx - img_x) / self.zoom
        pdf_y = (cy - img_y) / self.zoom

        best_idx = None
        best_dist = float("inf")

        for idx, rects in enumerate(self.chunk_pdf_rects):
            for x0, y0, x1, y1 in rects:
                dy = 0.0 if (y0 - 2 <= pdf_y <= y1 + 2) else min(abs(pdf_y - y0), abs(pdf_y - y1))
                dx = 0.0 if (x0 - 4 <= pdf_x <= x1 + 4) else min(abs(pdf_x - x0), abs(pdf_x - x1)) * 0.25
                dist = dy + dx
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        return best_idx if best_dist <= 28.0 else None

    def _on_textbox_click(self, event):
        idx = self._chunk_idx_from_textbox_xy(event.x, event.y)
        if idx is not None and not self.is_reading:
            self.select_chunk(idx, scroll_into_view=False, scroll_pdf=True)

    def _on_textbox_double_click(self, event):
        idx = self._chunk_idx_from_textbox_xy(event.x, event.y)
        if idx is not None:
            self.select_chunk(idx, scroll_into_view=False, scroll_pdf=True)
            self.start_reading(from_chunk=idx)
        return "break"

    def _on_pdf_canvas_single_click(self, event):
        if self.is_reading:
            return
        idx = self._chunk_idx_from_pdf_xy(event.x, event.y)
        if idx is not None:
            self.select_chunk(idx, scroll_into_view=True, scroll_pdf=False)

    def _on_pdf_canvas_double_click(self, event):
        idx = self._chunk_idx_from_pdf_xy(event.x, event.y)
        if idx is not None:
            self.select_chunk(idx, scroll_into_view=True, scroll_pdf=False)
            self.start_reading(from_chunk=idx)

    def select_chunk(self, idx: int, scroll_into_view: bool = True, scroll_pdf: bool = False):
        if not self.page_chunks:
            self.selected_chunk_idx = 0
            return

        idx = max(0, min(idx, len(self.page_chunks) - 1))
        self.selected_chunk_idx = idx

        self.text_box.tag_remove("selected_start", "1.0", tk.END)
        start_pos, end_pos = self.chunk_ranges[idx]
        self.text_box.tag_add("selected_start", start_pos, end_pos)
        if scroll_into_view and self.text_panel_visible:
            self.text_box.see(start_pos)

        if not self.is_reading:
            t = THEMES[self.theme_var.get()]
            self.status_lbl.config(
                text=f"Page {self.current_page + 1} • Ready at sentence {idx + 1}/{len(self.page_chunks)}",
                fg=t["status_fg"],
            )
            self._draw_pdf_sentence_highlight(idx, style="selected", auto_scroll=scroll_pdf)
            self.schedule_save_state()

    def step_sentence(self, delta: int):
        if not self.page_chunks:
            return
        new_idx = max(0, min(self.selected_chunk_idx + delta, len(self.page_chunks) - 1))
        if self.is_reading:
            self.start_reading(from_chunk=new_idx)
        else:
            self.select_chunk(new_idx, scroll_into_view=True, scroll_pdf=True)

    def restart_page_audio(self):
        if not self.doc or not self.page_chunks:
            return
        self.select_chunk(0, scroll_into_view=True, scroll_pdf=True)
        self.start_reading(from_chunk=0)

    # ================= PDF LOADING, BOOKMARK RESTORE & EXTRACTION =================

    def open_pdf(self):
        initial_dir = str(Path(self.current_pdf_path).parent) if self.current_pdf_path else str(Path.home())
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("PDF Files", "*.pdf")])
        if not path:
            return
        self.save_state_now()
        self._load_pdf_file(path, is_startup_restore=False)

    def _load_pdf_file(self, path: str, is_startup_restore: bool = False):
        norm_path = str(Path(path).resolve())
        self.stop_reading(preserve_position=False)
        self.ram_audio_cache.clear()

        try:
            self._restoring_session = True
            with self.doc_lock:
                if self.doc:
                    self.doc.close()
                self.doc = fitz.open(norm_path)
                total_p = len(self.doc)

            self.current_pdf_path = norm_path
            self.total_pages_lbl.config(text=f"/ {total_p}")
            self.root.title(f"Offline eBook Reader — {os.path.basename(norm_path)}")

            book_entry = self.saved_state.get("books", {}).get(norm_path, {})
            saved_page = int(book_entry.get("page", 0))
            saved_sentence = int(book_entry.get("sentence", 0))
            saved_zoom = book_entry.get("zoom", None)

            self.current_page = max(0, min(saved_page, total_p - 1))
            if isinstance(saved_zoom, (int, float)):
                self.zoom = max(0.6, min(3.0, round(float(saved_zoom), 2)))

            self.selected_chunk_idx = max(0, saved_sentence)
            self._populate_toc_tree()
            self.render_current_page(
                reset_chunk_idx=False,
                scroll_text_to_selection=True,
                scroll_pdf_to_selection=True,
            )

            if book_entry:
                t_fg = THEMES[self.theme_var.get()]["status_fg"]
                prefix = "Restored bookmark" if is_startup_restore else "Resumed bookmark"
                self.status_lbl.config(
                    text=f"{prefix}: Page {self.current_page + 1}, Sentence {self.selected_chunk_idx + 1}/{max(1, len(self.page_chunks))} ✓",
                    fg=t_fg,
                )
        except Exception as exc:
            if not is_startup_restore:
                messagebox.showerror("Error Opening PDF", str(exc))
        finally:
            self._restoring_session = False
            self.save_state_now()

    def get_total_pages(self) -> int:
        with self.doc_lock:
            return len(self.doc) if self.doc else 0

    def extract_page_chunks_and_rects(
        self, page_idx: int, skip_margins: bool
    ) -> tuple[list[str], list[list[tuple[float, float, float, float]]], set[int]]:
        with self.doc_lock:
            if not self.doc or not (0 <= page_idx < len(self.doc)):
                return [], [], set()
            page = self.doc[page_idx]
            rect = page.rect
            if skip_margins:
                margin_y = rect.height * 0.06
                clip_rect = fitz.Rect(rect.x0, rect.y0 + margin_y, rect.x1, rect.y1 - margin_y)
                text_dict = page.get_text("dict", clip=clip_rect, sort=True)
            else:
                text_dict = page.get_text("dict", sort=True)

        chunks: list[str] = []
        chunk_rects: list[list[tuple[float, float, float, float]]] = []
        para_end_indices: set[int] = set()

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            para_str = ""
            line_spans_map: list[tuple[int, int, tuple[float, float, float, float]]] = []

            for line in block.get("lines", []):
                lx0, ly0, lx1, ly1 = line["bbox"]
                raw_line = "".join(sp.get("text", "") for sp in line.get("spans", ""))
                raw_line = unicodedata.normalize("NFKC", raw_line)
                raw_line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200d\ufeff]", "", raw_line)
                raw_line = re.sub(r"[\u00ad\u2010\u2011]", "-", raw_line)
                raw_line = re.sub(r"[ \t]+", " ", raw_line).strip()
                if not raw_line:
                    continue

                if para_str.endswith("-") and len(para_str) >= 2 and para_str[-2].isalpha() and raw_line[0].isalpha():
                    para_str = para_str[:-1]
                    c_start = len(para_str)
                    para_str += raw_line
                else:
                    if para_str:
                        para_str += " "
                    c_start = len(para_str)
                    para_str += raw_line

                c_end = len(para_str)
                line_spans_map.append((c_start, c_end, (float(lx0), float(ly0), float(lx1), float(ly1))))

            if not para_str or not SPEAKABLE_RE.search(para_str):
                continue

            atomic_pieces: list[tuple[str, int, int]] = []
            for m in re.finditer(r"\S.*?(?:(?<=[.!?])(?=\s|$)|$)", para_str):
                seg_text = m.group(0).strip()
                if not seg_text:
                    continue
                s_off, e_off = m.start(), m.end()
                if len(seg_text) > 420:
                    for sm in re.finditer(r"\S.*?(?:(?<=[,;:])(?=\s|$)|$)", seg_text):
                        sub_t = sm.group(0).strip()
                        if sub_t:
                            atomic_pieces.append((sub_t, s_off + sm.start(), s_off + sm.end()))
                else:
                    atomic_pieces.append((seg_text, s_off, e_off))

            merged_para: list[tuple[str, int, int]] = []
            buf_text = ""
            buf_start = 0
            buf_end = 0

            for p_text, p_s, p_e in atomic_pieces:
                if not buf_text:
                    buf_text, buf_start, buf_end = p_text, p_s, p_e
                elif (ABBREV_END_RE.search(buf_text) or len(buf_text) < 25) and len(buf_text) + len(p_text) < 420:
                    buf_text = f"{buf_text} {p_text}"
                    buf_end = p_e
                else:
                    if SPEAKABLE_RE.search(buf_text):
                        merged_para.append((buf_text, buf_start, buf_end))
                    buf_text, buf_start, buf_end = p_text, p_s, p_e

            if buf_text and SPEAKABLE_RE.search(buf_text):
                merged_para.append((buf_text, buf_start, buf_end))

            for c_text, c_s, c_e in merged_para:
                matched_rects: list[tuple[float, float, float, float]] = []
                for ls, le, lbox in line_spans_map:
                    if ls < c_e and le > c_s:
                        line_chars = max(1, le - ls)
                        ov_s = max(c_s, ls)
                        ov_e = min(c_e, le)
                        lx0, ly0, lx1, ly1 = lbox
                        lw = lx1 - lx0
                        sub_x0 = lx0 + lw * ((ov_s - ls) / line_chars) if ov_s > ls + 1 else lx0
                        sub_x1 = lx0 + lw * ((ov_e - ls) / line_chars) if ov_e < le - 1 else lx1
                        matched_rects.append((sub_x0, ly0, sub_x1, ly1))

                chunks.append(c_text)
                chunk_rects.append(self._merge_adjacent_line_rects(matched_rects))

            if merged_para:
                para_end_indices.add(len(chunks) - 1)

        return chunks, chunk_rects, para_end_indices

    def render_current_page(
        self,
        reset_chunk_idx: bool = True,
        scroll_text_to_selection: bool = False,
        scroll_pdf_to_selection: bool = False,
    ):
        if not self.doc:
            return

        with self.doc_lock:
            page = self.doc[self.current_page]
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

        raw_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.base_themed_pil = self._apply_pdf_page_theme(raw_img)
        self.tk_image = ImageTk.PhotoImage(self.base_themed_pil)

        self.canvas.delete("all")
        canvas_w = max(self.canvas.winfo_width(), pix.width + 20)
        img_x = max(10, (canvas_w - pix.width) // 2)
        img_y = 10
        self.img_offset = (img_x, img_y)

        self.canvas_img_id = self.canvas.create_image(img_x, img_y, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, canvas_w, pix.height + 20))
        if not scroll_pdf_to_selection:
            self.canvas.yview_moveto(0)

        self.page_entry.delete(0, tk.END)
        self.page_entry.insert(0, str(self.current_page + 1))

        self._sync_toc_highlight_to_page()

        self.page_chunks, self.chunk_pdf_rects, para_breaks = self.extract_page_chunks_and_rects(
            self.current_page, self.skip_margins_var.get()
        )
        self.chunk_ranges.clear()

        self.text_box.config(state=tk.NORMAL)
        self.text_box.delete("1.0", tk.END)

        if not self.page_chunks:
            self.text_box.insert(tk.END, "[No speakable text found on this page]")
            self.selected_chunk_idx = 0
        else:
            for idx, chunk in enumerate(self.page_chunks):
                start_pos = self.text_box.index("end-1c")
                self.text_box.insert(tk.END, chunk, (f"chunk_{idx}",))
                end_pos = self.text_box.index("end-1c")
                self.chunk_ranges.append((start_pos, end_pos))

                if idx in para_breaks:
                    self.text_box.insert(tk.END, "\n\n")
                else:
                    self.text_box.insert(tk.END, " ")

            target_idx = 0 if reset_chunk_idx else min(self.selected_chunk_idx, len(self.page_chunks) - 1)
            self.select_chunk(
                target_idx,
                scroll_into_view=scroll_text_to_selection,
                scroll_pdf=scroll_pdf_to_selection,
            )

        self.text_box.config(state=tk.DISABLED)

    def prev_page(self):
        if self.doc and self.current_page > 0:
            was_reading = self.is_reading
            self.stop_reading(preserve_position=False)
            self.current_page -= 1
            self.render_current_page(reset_chunk_idx=True)
            self.schedule_save_state()
            if was_reading:
                self.start_reading(from_chunk=0)

    def next_page(self):
        if self.doc and self.current_page < self.get_total_pages() - 1:
            was_reading = self.is_reading
            self.stop_reading(preserve_position=False)
            self.current_page += 1
            self.render_current_page(reset_chunk_idx=True)
            self.schedule_save_state()
            if was_reading:
                self.start_reading(from_chunk=0)

    def jump_to_page(self, _event=None):
        if not self.doc:
            return
        try:
            target = int(self.page_entry.get()) - 1
            if 0 <= target < self.get_total_pages():
                self.stop_reading(preserve_position=False)
                self.current_page = target
                self.render_current_page(reset_chunk_idx=True)
                self.schedule_save_state()
                self.root.focus_set()
        except ValueError:
            pass

    def change_zoom(self, delta: float):
        if not self.doc:
            return
        self.zoom = max(0.6, min(3.0, round(self.zoom + delta, 2)))
        self.render_current_page(reset_chunk_idx=False, scroll_pdf_to_selection=True)
        if self.is_reading:
            self._highlight_active_chunk(self.selected_chunk_idx)
        self.schedule_save_state()

    # ================= OFFLINE KOKORO DOUBLE-BUFFER STREAMING =================

    def start_reading(self, from_chunk: int | None = None):
        if not self.doc:
            return
        if not self.model_ready or self.kokoro is None:
            messagebox.showinfo("Loading Model", "Kokoro ONNX is still loading or model files are missing.")
            return

        if from_chunk is not None:
            self.selected_chunk_idx = max(0, from_chunk)

        self.stop_reading(preserve_position=True)
        self.session_id += 1
        sid = self.session_id

        self.is_reading = True
        self.is_paused = False

        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
        self.stop_btn.config(state=tk.NORMAL)

        voice_id, lang = self._get_voice_and_lang()
        speed = self._get_speed_float()
        skip_m = self.skip_margins_var.get()

        threading.Thread(
            target=self._playback_coordinator,
            args=(sid, self.current_page, self.selected_chunk_idx, voice_id, lang, speed, skip_m),
            daemon=True,
        ).start()

    def toggle_pause(self):
        if not self.is_reading:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.status_lbl.config(
                text=f"Paused at sentence {self.selected_chunk_idx + 1}/{max(1, len(self.page_chunks))} (Bookmark saved ✓)"
            )
            self.save_state_now()
        else:
            self.pause_btn.config(text="⏸ Pause")

    def stop_reading(self, preserve_position: bool = True):
        self.session_id += 1
        self.is_reading = False
        self.is_paused = False

        self.text_box.tag_remove("speaking", "1.0", tk.END)
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)

        if preserve_position and self.page_chunks:
            self.select_chunk(self.selected_chunk_idx, scroll_into_view=False, scroll_pdf=False)
            self.status_lbl.config(
                text=f"Stopped at sentence {self.selected_chunk_idx + 1}/{len(self.page_chunks)} (Bookmark saved ✓)"
            )
            self.schedule_save_state()
        else:
            self._draw_pdf_sentence_highlight(None)

    def _synthesize_sentence_ram(
        self, page_idx: int, chunk_idx: int, text: str, voice_id: str, lang: str, speed: float, skip_m: bool
    ) -> tuple[np.ndarray, int] | None:
        cache_key = (page_idx, chunk_idx, voice_id, speed, skip_m)
        cached = self.ram_audio_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.kokoro is None or not SPEAKABLE_RE.search(text):
            return None

        with self.kokoro_lock:
            samples, sr = self.kokoro.create(text, voice=voice_id, speed=speed, lang=lang)

        samples_f32 = np.asarray(samples, dtype=np.float32)
        pause_pad = np.zeros(int(sr * 0.14), dtype=np.float32)
        full_pcm = np.concatenate([samples_f32, pause_pad])

        if len(self.ram_audio_cache) >= 40:
            self.ram_audio_cache.pop(next(iter(self.ram_audio_cache)), None)
        self.ram_audio_cache[cache_key] = (full_pcm, sr)
        return full_pcm, sr

    def _playback_coordinator(
        self,
        sid: int,
        start_page: int,
        start_chunk: int,
        voice_id: str,
        lang: str,
        speed: float,
        skip_m: bool,
    ):
        pcm_queue: queue.Queue[tuple[int, int, int, np.ndarray, int] | None] = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)

        def producer():
            total_pages = self.get_total_pages()
            p_idx = start_page
            try:
                while p_idx < total_pages and sid == self.session_id:
                    chunks, _rects, para_ends = self.extract_page_chunks_and_rects(p_idx, skip_m)
                    total_c = len(chunks)
                    c_start = start_chunk if p_idx == start_page else 0

                    for c_idx in range(c_start, total_c):
                        if sid != self.session_id:
                            return
                        res = self._synthesize_sentence_ram(
                            p_idx, c_idx, chunks[c_idx], voice_id, lang, speed, skip_m
                        )
                        if res is None:
                            continue
                        pcm, sr = res
                        if c_idx in para_ends:
                            pcm = np.concatenate([pcm, np.zeros(int(sr * 0.20), dtype=np.float32)])

                        while sid == self.session_id:
                            try:
                                pcm_queue.put((p_idx, c_idx, total_c, pcm, sr), timeout=0.1)
                                break
                            except queue.Full:
                                continue

                    if not self.auto_advance_var.get():
                        break
                    p_idx += 1
            except Exception as exc:
                if sid == self.session_id:
                    self.root.after(0, lambda e=exc: self.status_lbl.config(text=f"Synth Error: {e}", fg="#ef5350"))
            finally:
                try:
                    pcm_queue.put(None, timeout=0.2)
                except queue.Full:
                    pass

        threading.Thread(target=producer, daemon=True).start()

        stream: sd.OutputStream | None = None
        current_sr = 24000
        active_ui_page = start_page

        try:
            self.root.after(
                0, lambda: self.status_lbl.config(text="Synthesizing first sentence locally...", fg="#ffb74d")
            )

            while sid == self.session_id:
                try:
                    item = pcm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None or sid != self.session_id:
                    break

                p_idx, c_idx, total_c, pcm, sr = item

                if p_idx != active_ui_page:
                    if not self.auto_advance_var.get():
                        self.selected_chunk_idx = 0
                        break
                    active_ui_page = p_idx
                    turned_evt = threading.Event()

                    def _turn_page(target_p=p_idx):
                        self.current_page = target_p
                        self.render_current_page(reset_chunk_idx=True)
                        turned_evt.set()

                    self.root.after(0, _turn_page)
                    turned_evt.wait(timeout=2.0)

                if stream is None or sr != current_sr:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    current_sr = sr
                    stream = sd.OutputStream(samplerate=current_sr, channels=1, dtype="float32")
                    stream.start()

                self.selected_chunk_idx = c_idx
                q_ready = pcm_queue.qsize()
                t_fg = THEMES[self.theme_var.get()]["status_fg"]
                self.root.after(
                    0,
                    lambda p=p_idx, c=c_idx, tot=total_c, qr=q_ready, fg_col=t_fg: (
                        self.status_lbl.config(
                            text=f"Reading Page {p + 1} • Sentence {c + 1}/{tot}", fg=fg_col
                        ),
                        self.buffer_lbl.config(
                            text=f"Offline Buffer: {qr}/{PREFETCH_QUEUE_SIZE} ready ✓", fg="#66bb6a"
                        ),
                        self._highlight_active_chunk(c),
                        self.schedule_save_state(delay_ms=500),
                    ),
                )

                step_frames = int(current_sr * 0.04)
                cursor = 0
                n_frames = len(pcm)

                while cursor < n_frames and sid == self.session_id:
                    if self.is_paused:
                        time.sleep(0.04)
                        continue
                    end = min(cursor + step_frames, n_frames)
                    stream.write(pcm[cursor:end].reshape(-1, 1))
                    cursor = end

            if sid == self.session_id and not self.auto_advance_var.get():
                self.selected_chunk_idx = 0

        except Exception as exc:
            if sid == self.session_id:
                self.root.after(0, lambda e=exc: self.status_lbl.config(text=f"Audio Error: {e}", fg="#ef5350"))
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            if sid == self.session_id:
                self.root.after(0, lambda: self.stop_reading(preserve_position=True))

    def _highlight_active_chunk(self, chunk_idx: int):
        if chunk_idx < 0 or chunk_idx >= len(self.chunk_ranges):
            return
        self.text_box.tag_remove("speaking", "1.0", tk.END)
        self.text_box.tag_remove("selected_start", "1.0", tk.END)
        start_pos, end_pos = self.chunk_ranges[chunk_idx]
        self.text_box.tag_add("speaking", start_pos, end_pos)
        if self.text_panel_visible:
            self.text_box.see(start_pos)

        self._draw_pdf_sentence_highlight(chunk_idx, style="speaking", auto_scroll=True)

    def _on_close(self):
        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except Exception:
                pass
            self._save_after_id = None
        self.save_state_now()
        self.stop_reading(preserve_position=False)
        with self.doc_lock:
            if self.doc:
                self.doc.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OfflineKokoroPDFReader(root)
    root.mainloop()