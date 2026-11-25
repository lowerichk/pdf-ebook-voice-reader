# Offline PDF eBook & Audio Reader (`book_reader.py`)

A 100% offline desktop PDF eBook and audiobook reader built in Python. Powered locally by **Kokoro ONNX** neural text-to-speech, it turns any readable PDF into a natural-sounding audiobook with zero internet dependency, no cloud API keys, and gapless in-memory double-buffering.

---

## Features

- **100% Offline Neural TTS:** Uses local `kokoro-onnx` models and raw PCM streaming (`sounddevice`) for human-grade narration with zero network latency or dropped sentences.
- **RAM Double-Buffering:** Pre-synthesizes upcoming sentences in a background thread—even across page turns—for instant, gap-free playback.
- **Live Dual Sentence Highlighting:** Draws a soft translucent highlight box directly over the spoken lines on the rendered PDF page (auto-scrolling as it reads) alongside a clickable side text panel.
- **Click Anywhere to Read:** Single-click to select a starting sentence, or double-click any sentence (directly on the PDF page or in the side panel) to start reading immediately from that exact spot.
- **3 Reading Themes:** Switch between **Dark Mode** (with smart diagram/chart color preservation), **Sepia Mode** (warm paper), and **Light Mode**.
- **Collapsible Sidebars:** Toggle the **Table of Contents (TOC)** chapter tree (`C`) or hide the **Text Panel** (`T`) for a centered, distraction-free full-width PDF view.
- **Automatic Bookmarking & Recent Books:** Automatically remembers your last opened PDF, exact page, sentence index, zoom level, voice, speed, and layout in `reader_bookmarks.json`.

---

## System Requirements

- **Python:** Python **3.10 – 3.12** (3.12 recommended; requires `tkinter` included)
- **Operating System:** macOS (optimized for Apple Silicon M1/M2/M3/M4 & Intel), Windows 10/11, or Linux
- **RAM & Disk Space:** ~600 MB free RAM during synthesis; ~200 MB disk space for local Kokoro model weights
- **System Audio / Phonemizer Libraries (OS-specific):**
  - **macOS:** Works out of the box on most setups. If `tkinter`, `sounddevice`, or `espeak` ever report a missing system library:
    ```bash
    brew install python-tk portaudio espeak-ng
    ```
  - **Ubuntu / Debian Linux:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3-tk libportaudio2 libsndfile1 espeak-ng
    ```
  - **Windows:** Standard official Python installer from [python.org](https://www.python.org/) includes `tkinter` and binary wheels for all audio libraries automatically.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)lowerichk/pdf-ebook-voice-reader.git
cd <your-repo-name>
```

### 2. (Optional) Create & Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell / CMD)
```

### 3. Install Required Python Libraries
Install via `requirements.txt`:
```bash
pip install -r requirements.txt
```
Or install the packages directly:
```bash
pip install kokoro-onnx onnxruntime sounddevice soundfile numpy pymupdf pillow
```

### 4. Download the Local Kokoro ONNX Model Files (Required Once)
Because GitHub blocks large binary weights (`>100 MB`), download the Kokoro v1.0 model and voice pack directly into the same directory as `book_reader.py`:

```bash
# Option A: FP16 Model (156 MB — compact & fast)
curl -L -O [https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.fp16.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.fp16.onnx)

# Option B (Alternative): Native FP32 Model (310 MB — zero CPU type-casting overhead)
# curl -L -O [https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx)

# Voice Pack (27 MB — required for all models)
curl -L -O [https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin)
```

Once `kokoro-v1.0.fp16.onnx` (or `kokoro-v1.0.onnx`) and `voices-v1.0.bin` are inside the project folder, **no internet connection is ever needed again.**

---

## Running the Reader

```bash
python3 book_reader.py
```

1. Click **Open** in the top-left toolbar to load any text-based `.pdf` book.
2. Double-click any sentence on the PDF page (or click **▶ Play** / press **Spacebar**) to start listening.
3. Close the app anytime—reopening `book_reader.py` will automatically restore your book, page, sentence, zoom, and theme.

---

## Keyboard Shortcuts & Controls

| Shortcut / Action | Function |
| :--- | :--- |
| **`Spacebar`** | Play / Pause audio narration |
| **`←` / `→`** | Previous / Next PDF page |
| **`C`** | Show / Hide the **Table of Contents (TOC)** chapter sidebar |
| **`T`** | Show / Hide the right-hand **Text Panel** (full-width PDF mode) |
| **`D`** | Cycle reading theme (**Dark** → **Sepia** → **Light**) |
| **Single-Click Sentence** | Set starting sentence (works on both PDF page & right text panel) |
| **Double-Click Sentence** | Immediately play narration starting from that exact sentence |

---

## Project Structure

```text
.
├── book_reader.py            # Main desktop PDF eBook & offline TTS application
├── requirements.txt          # Python package dependencies
├── .gitignore                # Excludes large .onnx/.bin weights, PDFs, and local bookmarks
├── kokoro-v1.0.fp16.onnx     # Local Kokoro neural TTS model (downloaded locally)
├── voices-v1.0.bin           # Local Kokoro voice embeddings (downloaded locally)
└── reader_bookmarks.json     # Auto-generated local reading progress & preferences
```

---

## Troubleshooting

- **"No speakable text found on this page":** Your PDF is likely made of scanned page images rather than embedded digital text. Run the PDF through an OCR tool (such as `ocrmypdf input.pdf output.pdf`) first to add a selectable text layer.
- **macOS Window Opens Behind Terminal:** If launching from Terminal, `book_reader.py` automatically lifts the window to the front; if macOS Focus steal-prevention blocks it, click the Python icon in your Dock.
- **`PortAudio library not found` on macOS/Linux:** Install the native PortAudio library (`brew install portaudio` on macOS or `sudo apt install libportaudio2` on Linux).