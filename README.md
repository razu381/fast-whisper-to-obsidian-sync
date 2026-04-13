# fast-whisper-to-obsidian-sync
# Whisper Live Transcribe & Obsidian Sync

This repository contains a suite of local, AI-powered transcription scripts that hook into global Linux keyboard shortcuts to instantly transcribe voice to text using `faster_whisper`.

The tools are designed primarily to integrate with an Obsidian vault ("My notes"), syncing automatically to Git, but also support system-wide cursor dictation.

## 🚀 Features & Keyboard Shortcuts

Based on your GNOME custom keybinding configuration (`setup.sh`), hitting these shortcuts spawns a GTK-based animated recording widget over your screen.

- **`Ctrl+Shift+J` (Journal Mode):** 
  Transcribes and appends text directly to today's daily journal file (e.g., `Daily journal/YYYY-MM-DD.md`) with timestamps.
- **`Ctrl+Shift+I` (Inbox / Idea Mode):**
  Creates a brand new markdown file under `Inbox/` (e.g. `YYYY-MM-DD-idea-HH-MM.md`) containing the transcription. Great for sudden dumps of knowledge.
- **`Ctrl+Shift+S` (System-Wide Dictation):**
  Uses `xdotool` to synthesize keystrokes, effectively pasting your speech directly into whichever window or input field currently has focus.

## 🧠 Architecture & File Structure

- **`whisper_base.py`:** The core engine that powers the UI, audio recording, background transcription, locking mechanism, and Git commits. All shortcut wrappers call `run(mode)` from this file.
- **`transcribe_journal.py`:** Wrapper script for the Journal mode.
- **`transcribe_inbox.py`:** Wrapper script for the Inbox mode.
- **`transcribe_system.py`:** Wrapper script for the System mode.
- **`setup.sh`:** Bash script to install system dependencies (`xdotool`, `gir1.2-gtk-3.0`), register the GNOME desktop shortcuts, and configure the Python execution.
- **Venv Setup (`~/whisper-env/`):** Python dependencies are isolated in a virtual environment (`faster-whisper`, `numpy`, `pyaudio`). However, we inject the system's `dist-packages` inside `whisper_base.py` to allow the Venv to utilize system GTK dependencies (PyGObject).
- **`whisper_debug.log`:** A rotating debug log file (stored alongside these scripts, outside the vault) that tracks session start failures, unhandled thread exceptions, and success markers. It retains 5 backups (1MB each) for local reporting.

## ⚡ Technical Mechanics Explained (For AI Reference)

If you are an AI assistant tasked with modifying these tools later, keep the following mechanisms in mind:

### 1. Process Lifecycle & Single-Instance Locking
Because users can trigger shortcuts aggressively, the app employs a **graceful-takeover lock** via `/tmp/whisper_active.pid`.
- If a new shortcut is pressed while an old instance is running, the new instance sends `SIGTERM` to the old process and waits up to 12 seconds.
- The old instance traps the `SIGTERM` via `GLib.unix_signal_add` inside the GTK main loop, stops recording, drains the remaining audio queue, saves the data to Markdown, and completes its background Git push before cleanly dying.

### 2. The Spinner "Saving" State
Because transcription on a CPU might lag slightly behind real-time speech, the GTK Window doesn't violently destruct upon hitting "Stop". It transitions to a `is_saving = True` state:
- The UI replaces the voice waveform with a green spinning arc.
- The PyAudio stream yields, but the background thread continues pulling chunks from `audio_queue` until empty.
- Once completely drained, `_git_commit_sync` is invoked blocking on the main thread to secure the Markdown data before `Gtk.main_quit()` cleanly triggers.

### 3. Git Automation (Synchronous Background Task)
File saving and `git commit` operate synchronously to guarantee data safety immediately when transcription halts. However, `git push` takes network overhead. Instead of freezing the GTK main loop, the save operations are offloaded into a dedicated `threading.Thread` (daemon=False) spawned behind the scenes. 

The main GTK Window keeps its "Saving..." spinner active until this thread fully succeeds pushing data securely to the origin remote (or times out/fails). Only then does the thread issue `Gtk.main_quit()` cleanly, ensuring cloud synchronization never fails silently from abrupt process cancellation.

### 4. Continuous Flow Audio
The system uses `pyaudio` polling frames into a thread-safe `queue.Queue()`. Audio vectors are aggregated, normalized to floats (`/ 32768.0`), aggressively pruned for silence (`np.abs(audio).mean() < 0.005`), and handed to the Int8 CPU-based Whisper medium model.
