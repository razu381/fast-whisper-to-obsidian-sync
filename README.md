# fast-whisper-to-obsidian-sync
# Whisper Live Transcribe & Obsidian Sync

This repository contains a suite of local, AI-powered transcription scripts that hook into global Linux keyboard shortcuts to instantly transcribe voice to text using `faster_whisper`.

The tools are designed primarily to integrate with an Obsidian vault ("My notes"), syncing automatically to Git, but also support system-wide cursor dictation.

## 🚀 Features & Keyboard Shortcuts

Based on your GNOME custom keybinding configuration (`setup.sh`), hitting these shortcuts spawns a GTK-based animated recording widget over your screen.

- **`Ctrl+Shift+J` (Journal Mode):** 
  Transcribes and appends text directly to today's daily journal file (e.g., `Planner/Daily/YYYY-MM-DD.md`) inside the `## 🎙 Voice Capture` section.
- **`Ctrl+Shift+I` (Inbox / Idea Mode):**
  Creates a brand new markdown file under `Inbox/` (e.g. `YYYY-MM-DD-idea-HH-MM.md`) containing the transcription. Great for sudden knowledge dumps.
- **`Ctrl+Shift+S` (System-Wide Dictation):**
  Uses `xdotool` to synthesize keystrokes, pasting your speech directly into whichever window or input field currently has focus.

## 🧠 Architecture & File Structure

- **`whisper_base.py`:** The core engine — UI, audio recording, background transcription, post-processing pipeline, locking, and Git commits. All shortcut wrappers call `run(mode)` from this file.
- **`transcribe_journal.py`:** Wrapper script for Journal mode.
- **`transcribe_inbox.py`:** Wrapper script for Inbox mode.
- **`transcribe_system.py`:** Wrapper script for System-Wide Dictation mode.
- **`transcribe_book.py`:** Wrapper script for Book Notes mode (3-layer fuzzy matching to the correct book file).
- **`setup.sh`:** Installs system dependencies (`xdotool`, `gir1.2-gtk-3.0`), registers GNOME shortcuts, and configures Python execution.
- **`setup_gemini_key.sh`:** One-time helper to safely save your Gemini API key to `~/.bashrc`.
- **`~/whisper-env/`:** Isolated Python venv (`faster-whisper`, `numpy`, `pyaudio`, `google-generativeai`). System `dist-packages` are injected for PyGObject/GTK access.
- **`whisper_debug.log`:** Rotating debug log (outside the vault) — tracks session starts, errors, and success markers. 5 backups × 1 MB each.

## ✨ Transcription Quality Pipeline

Each audio chunk flows through a 4-stage post-processing pipeline before being written to Obsidian:

```
Raw Whisper output
  → _smart_join()         # sentence-aware line breaks (no more wall-of-text)
  → _apply_corrections()  # regex word-correction map (Fiverr, WooCommerce, etc.)
  → _gemini_polish()      # Gemini Flash: professional punctuation & paragraph formatting
  → written to Obsidian
```

### Quality Improvements Applied
| Improvement | Effect |
|---|---|
| `INITIAL_PROMPT` rewritten as natural sentences | Primes Whisper to reproduce correct spellings (Fiverr, WooCommerce, etc.) |
| `temperature=0.0` | Fully deterministic output — eliminates punctuation randomness |
| `vad_parameters` min silence 500ms | Full sentences finish before chunk cuts |
| `compression_ratio_threshold=2.4` | Rejects looping/repetitive output |
| `no_speech_threshold=0.6` | Stronger silence filter — fewer phantom words |
| `_CORRECTIONS` regex map | Guaranteed word-level fixes regardless of model output |
| `_smart_join()` | Natural line breaks between sentences |
| Gemini Flash post-processing | Near-perfect punctuation and paragraphs, free tier (1500 req/day) |
| `CHUNK_SECS = 20` | More context per call → better sentence-level accuracy |

### Expanding the Correction Map
If you spot a mis-transcription in your Obsidian notes, add it to `_CORRECTIONS` in `whisper_base.py`:
```python
_CORRECTIONS = [
    (r'\byour_wrong_word\b', 'CorrectWord'),
    ...
]
```

## ⚙️ First-Time Setup

### 1. Install dependencies
```bash
bash setup.sh
```

### 2. Install Python packages into the venv
```bash
/home/razu/whisper-env/bin/pip install faster-whisper pyaudio numpy google-generativeai
```

### 3. Set up your Gemini API key (free)
Get a free key at **https://aistudio.google.com/app/apikey**, then:
```bash
bash setup_gemini_key.sh YOUR_API_KEY_HERE
source ~/.bashrc
```
> **Security note:** The key is stored only in `~/.bashrc` — never inside this repo. `whisper_base.py` reads it via `os.environ.get("GEMINI_API_KEY", "")`. Your key will **never** appear in Git history.

### 4. Test Gemini connection
```bash
/home/razu/whisper-env/bin/python3 -c "
import google.generativeai as g, os
g.configure(api_key=os.environ['GEMINI_API_KEY'])
print(g.GenerativeModel('gemini-2.0-flash').generate_content('say: connected').text)
"
```

## ⚡ Technical Mechanics (For AI Reference)

If you are an AI assistant tasked with modifying these tools later, keep the following in mind:

### 1. Process Lifecycle & Single-Instance Locking
The app uses a **graceful-takeover lock** via `/tmp/whisper_active.pid`.
- A new shortcut press sends `SIGTERM` to any existing instance and waits up to 12 seconds.
- The old instance traps `SIGTERM` via `GLib.unix_signal_add`, stops recording, drains the audio queue, saves data to Markdown, and completes the Git push before exiting cleanly.

### 2. The "Saving" Spinner State
When the user clicks Stop, the GTK window transitions to `is_saving = True`:
- The UI shows a green spinning arc.
- The PyAudio stream stops, but the background thread keeps pulling chunks from `audio_queue` until empty.
- Only after the queue drains does `_git_commit_sync` run, then `Gtk.main_quit()` — guaranteeing no data loss.

### 3. Git Automation
File saving and `git commit` run synchronously for data safety. `git push` is offloaded to a `threading.Thread` (daemon=False) so it doesn't freeze the GTK loop. The "Saving..." spinner stays active until the push succeeds or times out.

### 4. Post-Processing Pipeline Order
`_smart_join()` → `_apply_corrections()` → `_gemini_polish()`. Each step is a pure function that returns a string. `_gemini_polish()` silently falls back to its input on any error (no key, network issue, rate limit) — the app never crashes due to the Gemini step.

### 5. Continuous Flow Audio
`pyaudio` polls frames into a thread-safe `queue.Queue()`. Audio is normalised to float32 (`/ 32768.0`), pruned for silence (`np.abs(audio).mean() < 0.005`), then passed to Whisper in 20-second chunks.

### 6. Book Mode — 3-Layer Matching
`transcribe_book.py` uses `_match_book()`: (1) number normalisation, (2) token Jaccard similarity on filenames, (3) Obsidian frontmatter aliases. A match confidence > 0.15 routes the note to the correct book file; otherwise it goes to `Unsorted_Notes.md`.
