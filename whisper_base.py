#!/home/razu/whisper-env/bin/python3
"""
whisper_base.py
Shared base for all whisper transcription scripts.
Shows countdown timer while model loads, then animated waveform when ready.
"""

import sys as _sys
# gi (PyGObject) lives in system packages; expose it even inside this venv
_sys.path.insert(0, "/usr/lib/python3/dist-packages")

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, Gdk
import pyaudio
import numpy as np
import threading
import queue
import os
import sys
import subprocess
import signal
from faster_whisper import WhisperModel
from datetime import datetime
import math
import time
import logging
import logging.handlers
import re
import difflib

# ── config ────────────────────────────────────────────
VAULT_PATH   = "/home/razu/Courses/Notes/My notes"
MODEL_SIZE   = "large-v3"    # Using the latest large model for best overall accuracy
LANGUAGE     = "en"
CHUNK_SECS   = 20   # Increased from 12 → full sentences fit in one transcription call
SAMPLE_RATE  = 16000
LOAD_SECONDS = 30   # estimated model load time for countdown

# Provide common vocabulary (jargon, names) so Whisper knows how to spell them!
# Add your own custom words separated by commas.
# Contextual sentences are FAR more effective than a word list.
# Whisper's language model sees these as preceding speech — it will
# copy the exact spellings and style into its output.
INITIAL_PROMPT = (
    "I work as a freelance web developer on Fiverr, building WooCommerce stores "
    "with Elementor Pro, JetEngine, and WPFunnels for clients. "
    "For headless WordPress projects I use Next.js, TypeScript, TailwindCSS, "
    "GSAP, Framer Motion, Shadcn, Astro, and Turbopack. "
    "My workflow includes GitHub Actions, Playwright, Lovable, Claude Code, and MCP. "
    "I organise everything in my Obsidian vault and track brands like "
    "ThirdEyeRebels, BebaCo, Haramain Apparels, and regalwatchbd. "
    "I read books by Alex Hormozi and study Health Informatics in Noakhali. "
    "I measure progress with metrics like CTR, Success Score, and Level Zero. "
    "Daily tools include JetBrains Mono, Ubuntu, Linux, GTK, Git, Pluckeye, "
    "ThemeForest, Envato, Creative Market, UI8, Gumroad, and Lara Casta."
)
# ──────────────────────────────────────────────────────

# ── Gemini Flash post-processing ──────────────────────
# Get a free key at https://aistudio.google.com/app/apikey
# Set GEMINI_API_KEY="..." in ~/.bashrc or pass via env before running.
GEMINI_ENABLED = True
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_PROMPT  = (
    "You are a professional editor. Fix the punctuation, capitalisation, and "
    "add natural paragraph breaks to this voice transcription. "
    "Clean up the speech by removing filler words (like 'um', 'ah', 'like'), "
    "false starts, and stuttering. "
    "Otherwise, preserve the meaning and vocabulary exactly — do NOT rephrase or summarise. "
    "Return ONLY the corrected text, nothing else. "
    "Correct these domain terms if mis-spelled: "
    "Fiverr, WooCommerce, Elementor, JetEngine, WordPress, Obsidian, GitHub, "
    "TailwindCSS, TypeScript, Framer Motion, GSAP, Shadcn, Astro, Turbopack, "
    "Playwright, ThirdEyeRebels, BebaCo, regalwatchbd, Haramain, Noakhali.\n\n"
    "Transcript:\n{text}"
)
# ──────────────────────────────────────────────────────

RECOVERY_FILE = os.path.expanduser("~/.whisper_recovery.txt")

# ── Deterministic word-correction map ─────────────────
# Catches stubborn mis-transcriptions that bypass the prompt.
# Add new patterns here as you discover them in your notes.
_CORRECTIONS = [
    # Fiverr mis-hearings
    (r'\bfiver\b',        'Fiverr'),
    (r'\bfibre\b',        'Fiverr'),
    (r'\bfever r\b',      'Fiverr'),
    (r'\bfive r\b',       'Fiverr'),
    (r'\bfree var\b',     'Fiverr'),
    # Platform / tool names
    (r'\bwoo commerce\b', 'WooCommerce'),
    (r'\bwoo-commerce\b', 'WooCommerce'),
    (r'\bword press\b',   'WordPress'),
    (r'\bjet engine\b',   'JetEngine'),
    (r'\bgit hub\b',      'GitHub'),
    (r'\btailwind css\b', 'TailwindCSS'),
    (r'\belementor pro\b','Elementor Pro'),
    (r'\bnoakhali\b',     'Noakhali'),
]

def _apply_corrections(text: str) -> str:
    """Apply regex-based deterministic word corrections to raw Whisper output."""
    for pattern, replacement in _CORRECTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _smart_join(segments_list) -> str:
    """
    Join Whisper segments into readable paragraphs.
    Segments ending with sentence-terminal punctuation (.!?) get a line break.
    Short trailing fragments are kept on the same line as the previous sentence.
    """
    lines  = []
    buffer = []
    for seg in segments_list:
        t = seg.text.strip()
        if not t:
            continue
        buffer.append(t)
        if t[-1] in '.!?':
            lines.append(' '.join(buffer))
            buffer = []
    if buffer:
        lines.append(' '.join(buffer))
    return '\n'.join(lines)


def _gemini_polish(text: str) -> str:
    """
    Send raw transcription text to Gemini Flash for punctuation and
    paragraph formatting. Falls back to raw text on any error.
    Requires GEMINI_API_KEY env var and the google-genai package.
    """
    if not GEMINI_ENABLED or not GEMINI_API_KEY:
        return text
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model_g = genai.GenerativeModel(GEMINI_MODEL)
        prompt  = GEMINI_PROMPT.format(text=text)
        response = model_g.generate_content(prompt)
        polished = response.text.strip()
        return polished if polished else text
    except Exception:
        logger.warning("Gemini polish failed — using raw corrected text", exc_info=True)
        return text


def _gemini_polish_safe(text: str) -> str:
    if not text or not text.strip():
        return text
    polished = _gemini_polish(text)
    if not polished or not polished.strip():
        return text
    if len(polished.strip()) < len(text.strip()) * 0.3:
        logger.warning("Gemini response too short — using raw corrected text")
        return text
    return polished


LOCK_FILE = "/tmp/whisper_active.pid"

JOURNAL_DIR   = os.path.join(VAULT_PATH, "Planner/Daily")
JOURNAL_TPL   = os.path.join(VAULT_PATH, "Planner/Templates/daily.md")
INBOX_DIR     = os.path.join(VAULT_PATH, "Inbox")
BOOKS_DIR     = os.path.join(VAULT_PATH, "Books")

# ── logging setup ─────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(VAULT_PATH), "whisper_debug.log")
logger = logging.getLogger("whisper_app")
logger.setLevel(logging.DEBUG)
# Keep up to 5 backups, 1MB each
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=1024*1024, backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
logger.info("--- New Whisper Session Started ---")
# ──────────────────────────────────────────────────────

# suppress ALSA/Jack noise
devnull = open(os.devnull, 'w')
os.dup2(devnull.fileno(), sys.stderr.fileno())

def notify(title, msg, timeout=4000):
    subprocess.Popen(["notify-send", "-t", str(timeout), title, msg])


def _check_recovery_file():
    if not os.path.exists(RECOVERY_FILE):
        return
    try:
        with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            os.remove(RECOVERY_FILE)
            return
        os.makedirs(INBOX_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-recovered-%H-%M")
        out = os.path.join(INBOX_DIR, f"{ts}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# Recovered Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{content}")
        os.remove(RECOVERY_FILE)
        notify("Whisper Recovery", "Unsaved session recovered and saved to Inbox.")
        logger.info(f"Recovery file restored to {out}")
    except Exception:
        logger.error("Recovery file check failed", exc_info=True)


# ── single-instance lock ──────────────────────────────
def _kill_existing():
    """Terminate any running whisper session and wait for it to exit."""
    if not os.path.exists(LOCK_FILE):
        return
    try:
        with open(LOCK_FILE) as f:
            old_pid = int(f.read().strip())
        if old_pid == os.getpid():
            return
        # Politely ask it to stop (SIGTERM triggers its save-and-exit handler)
        os.kill(old_pid, signal.SIGTERM)
        # Wait up to 12s for it to finish transcribing its remaining audio
        for _ in range(120):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)   # 0 = just check alive
            except ProcessLookupError:
                return                # gone — we're done
        # Still alive — force kill
        try:
            os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (ValueError, ProcessLookupError, FileNotFoundError, OSError):
        pass

def _write_lock():
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def _release_lock():
    try:
        with open(LOCK_FILE) as f:
            if int(f.read().strip()) == os.getpid():
                os.remove(LOCK_FILE)
    except (FileNotFoundError, ValueError, OSError):
        pass

# ── Layer 1: Number Normalization ─────────────────────
# Maps spoken/written number words to digits so that
# "hundred million" and "100M" both become "100000000".
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULTIPLIERS = {
    "hundred": 100, "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
}
_STOPWORDS = {"the", "a", "an", "by", "of", "in", "and", "to", "for", "is",
              "it", "my", "on", "at", "so", "i", "we", "he", "she", "they",
              "this", "that", "was", "with", "from", "as", "but", "or", "not",
              "how", "what", "which", "who", "where", "when", "do", "does"}

def _normalize_number_token(token):
    """Try to convert a single token like '100m' or '$200k' into a pure digit string."""
    clean = token.strip("$,.")
    # Already a pure number
    if clean.isdigit():
        return clean
    # Pattern like "100m", "25k", "1b"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([mkb])$', clean, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        suffix = m.group(2).lower()
        return str(int(num * _MULTIPLIERS.get(suffix, 1)))
    return None

def _normalize_text(text):
    """Normalize a text string: expand spoken numbers, strip punctuation, lowercase."""
    text = text.lower()
    # Pre-process: collapse comma-separated numbers like "100,000,000" → "100000000"
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Pre-process: collapse dollar signs into the number
    text = text.replace('$', '')
    text = re.sub(r'[^\w\s]', ' ', text)  # strip remaining punctuation
    words = text.split()

    # Pass 1: Convert word-numbers and digit+multiplier combos to canonical digits
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        # Check for standalone number token like "100m", "25k"
        num_tok = _normalize_number_token(w)
        if num_tok:
            # Peek ahead: is the next word a multiplier? e.g. "100 million"
            if i + 1 < len(words) and words[i+1] in _MULTIPLIERS:
                mult = _MULTIPLIERS[words[i+1]]
                result.append(str(int(float(num_tok)) * mult))
                i += 2
            else:
                result.append(num_tok)
                i += 1
            continue
        # Check if this starts a spoken number sequence like "one hundred million"
        if w in _WORD_TO_NUM or w in _MULTIPLIERS:
            total = 0
            current = 0
            while i < len(words) and (words[i] in _WORD_TO_NUM or words[i] in _MULTIPLIERS):
                w2 = words[i]
                if w2 in _WORD_TO_NUM:
                    current += _WORD_TO_NUM[w2]
                elif w2 in _MULTIPLIERS:
                    mult = _MULTIPLIERS[w2]
                    if current == 0:
                        current = 1
                    if mult >= 1000:
                        total += current * mult
                        current = 0
                    else:
                        current *= mult
                i += 1
            total += current
            if total > 0:
                result.append(str(total))
        else:
            result.append(w)
            i += 1
    return result

def _tokenize(text):
    """Normalize text and remove stopwords for comparison."""
    tokens = _normalize_text(text)
    return set(t for t in tokens if t not in _STOPWORDS and len(t) > 1)

def _jaccard(set_a, set_b):
    """Jaccard similarity between two sets of tokens."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

def _match_book(spoken_text, books_dir):
    """
    3-Layer local book matching engine.
    Returns (book_filename, confidence) or (None, 0.0).

    Layer 1: Number normalization (100M == hundred million == $100,000,000)
    Layer 2: Token-based Jaccard similarity with fuzzy word matching
    Layer 3: Obsidian frontmatter aliases
    """
    books = [f for f in os.listdir(books_dir) if f.endswith('.md') and f != "Unsorted_Notes.md"]
    if not books:
        return None, 0.0

    # Only evaluate the first 15 words, otherwise long dictations dilute the Jaccard score heavily
    intro_text = " ".join(spoken_text.split()[:15])
    spoken_tokens = _tokenize(intro_text)
    logger.debug(f"Book match — spoken tokens: {spoken_tokens}")

    best_score = 0.0
    best_book = None

    for book_file in books:
        book_name = book_file[:-3]  # strip .md

        # ── Layer 2: Token Jaccard on filename ───────
        file_tokens = _tokenize(book_name)
        score = _jaccard(spoken_tokens, file_tokens)

        # Boost: matching large numbers is a very strong signal
        for token in (spoken_tokens & file_tokens):
            if token.isdigit() and len(token) >= 4:
                score += 0.35  # heavy bonus for matching big numbers

        # Boost: also check fuzzy individual word matches
        # e.g. "offers" vs "offer" should still count
        for st in spoken_tokens:
            for ft in file_tokens:
                word_ratio = difflib.SequenceMatcher(None, st, ft).ratio()
                if word_ratio > 0.8 and st != ft:
                    score += 0.15  # bonus for close-enough words

        logger.debug(f"  {book_file}: filename score={score:.2f} (tokens={file_tokens})")

        # ── Layer 3: Obsidian aliases ────────────────
        try:
            with open(os.path.join(books_dir, book_file), "r", encoding="utf-8") as bf:
                content = bf.read(1024)
                alias_match = re.search(r'aliases:\s*\[(.*?)\]', content)
                if alias_match:
                    for alias in alias_match.group(1).split(','):
                        alias = alias.strip().strip("'\"")
                        if not alias:
                            continue
                        alias_tokens = _tokenize(alias)
                        alias_score = _jaccard(spoken_tokens, alias_tokens)
                        # Also fuzzy-boost alias words
                        for st in spoken_tokens:
                            for at in alias_tokens:
                                if difflib.SequenceMatcher(None, st, at).ratio() > 0.8 and st != at:
                                    alias_score += 0.15
                        logger.debug(f"    alias '{alias}': score={alias_score:.2f}")
                        score = max(score, alias_score)
        except Exception:
            pass

        if score > best_score:
            best_score = score
            best_book = book_file

    logger.info(f"Book match result: '{best_book}' with confidence {best_score:.2f}")
    return best_book, best_score

def _insert_into_voice_capture(filepath, text):
    """
    Append transcription text into the ## 🎙 Voice Capture section of a daily note.
    Inserts before the closing --- of that section (or at end of file if none found).
    If the section doesn't exist, falls back to appending at end of file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the Voice Capture heading line
        vc_idx = None
        for i, line in enumerate(lines):
            if re.match(r"^##\s+🎙\s+Voice Capture", line.strip()):
                vc_idx = i
                break

        if vc_idx is None:
            # Section not found — fall back to appending
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"\n{text}")
            return

        # Find the next --- separator after the Voice Capture heading
        insert_before = len(lines)  # default: end of file
        for i in range(vc_idx + 1, len(lines)):
            if lines[i].strip() == "---":
                insert_before = i
                break

        # Build the new line(s) to insert
        new_line = f"{text}\n"

        # Check if there's already content between heading and ---
        # (skip blank lines when checking)
        content_between = [
            l for l in lines[vc_idx + 1:insert_before] if l.strip()
        ]
        if content_between:
            # Append after existing content, before the ---
            lines.insert(insert_before, new_line)
        else:
            # Insert right after the heading (+ blank line)
            lines.insert(vc_idx + 1, new_line)

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        logger.error("Error inserting into Voice Capture section", exc_info=True)
        # Last-resort fallback
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n{text}")


def _git_commit_sync(mode, out_path):
    """Synchronous git add+commit; returns True if anything was committed."""
    try:
        subprocess.run(["git", "-C", VAULT_PATH, "add", "."],
                       capture_output=True, timeout=8)
        result = subprocess.run(
            ["git", "-C", VAULT_PATH, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5)
        if not result.stdout.strip():
            return False
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
        if mode == "journal":
            msg = f"auto: journal {ts}"
        elif mode == "book":
            msg = f"auto: book note {ts} — {os.path.basename(out_path) if out_path else 'unsorted'}"
        else:
            msg = f"auto: inbox {ts} — {os.path.basename(out_path)}"
        subprocess.run(["git", "-C", VAULT_PATH, "commit", "-m", msg],
                       capture_output=True, timeout=10)
        return True
    except Exception:
        return False




# ── GTK window ────────────────────────────────────────
class WhisperWindow(Gtk.Window):
    def __init__(self, mode, on_stop):
        super().__init__()
        self.on_stop     = on_stop
        self.mode        = mode
        self.phase       = 0.0
        self.is_ready    = False
        self.is_saving   = False
        self.chunks      = 0
        self.load_start  = time.time()
        self.countdown   = LOAD_SECONDS
        # Live microphone amplitude fed by _level_monitor thread (0.0=silent, 1.0=loud)
        self.audio_level = 0.0

        icon = {"journal": "📔", "inbox": "💡", "system": "🎙"}.get(mode, "🎙")
        mode_name = {"journal": "Journal", "inbox": "Inbox",
                     "system": "Dictate"}.get(mode, mode.capitalize())
        self.set_title(f"Whisper — {mode_name}")
        self.icon_str = icon

        self.set_default_size(340, 240)
        self.set_resizable(False)
        self.set_keep_above(True)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", self._on_close)

        # In system (dictate) mode, do NOT steal keyboard focus from the
        # target application — otherwise xdotool types into the Whisper window.
        if mode == "system":
            self.set_accept_focus(False)

        # styling
        css = b"""
        window          { background-color: #1e1e2e; }
        label           { color: #cdd6f4; font-family: sans-serif; }
        #lbl-title      { font-size: 17px; font-weight: bold; color: #cba6f7; }
        #lbl-countdown  { font-size: 28px; font-weight: bold; color: #f38ba8; }
        #lbl-hint       { font-size: 12px; color: #585b70; }
        #lbl-status     { font-size: 13px; color: #a6e3a1; }
        #lbl-chunks     { font-size: 12px; color: #6c7086; }
        button          { background: #313244; color: #cdd6f4; border: none;
                          border-radius: 8px; padding: 8px 28px; font-size: 13px; }
        button:hover    { background: #45475a; }
        button:disabled { background: #181825; color: #45475a; }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(22)
        vbox.set_margin_bottom(22)
        vbox.set_margin_start(24)
        vbox.set_margin_end(24)
        self.add(vbox)

        # title
        self.lbl_title = Gtk.Label(label=f"{icon} Loading model...")
        self.lbl_title.set_name("lbl-title")
        vbox.pack_start(self.lbl_title, False, False, 0)

        # canvas (countdown circle OR waveform)
        self.canvas = Gtk.DrawingArea()
        self.canvas.set_size_request(292, 80)
        self.canvas.connect("draw", self._draw)
        vbox.pack_start(self.canvas, False, False, 0)

        # countdown label (hidden once ready)
        self.lbl_countdown = Gtk.Label()
        self.lbl_countdown.set_name("lbl-countdown")
        self.lbl_countdown.set_text(f"~{self.countdown}s")
        vbox.pack_start(self.lbl_countdown, False, False, 0)

        # hint
        self.lbl_hint = Gtk.Label(label="Model loading — you can speak as soon as timer hits 0")
        self.lbl_hint.set_name("lbl-hint")
        self.lbl_hint.set_line_wrap(True)
        self.lbl_hint.set_justify(Gtk.Justification.CENTER)
        vbox.pack_start(self.lbl_hint, False, False, 0)

        # status (hidden until ready)
        self.lbl_status = Gtk.Label(label="")
        self.lbl_status.set_name("lbl-status")
        vbox.pack_start(self.lbl_status, False, False, 0)

        # chunk counter
        self.lbl_chunks = Gtk.Label(label="")
        self.lbl_chunks.set_name("lbl-chunks")
        vbox.pack_start(self.lbl_chunks, False, False, 0)

        # button
        self.btn = Gtk.Button(label="Stop & Save")
        self.btn.connect("clicked", self._on_close)
        self.btn.set_sensitive(False)
        vbox.pack_start(self.btn, False, False, 0)

        self.show_all()
        GLib.timeout_add(50, self._tick)      # 20fps animation
        GLib.timeout_add(1000, self._tick_countdown)  # 1s countdown

    # ── countdown tick ────────────────────────────────
    def _tick_countdown(self):
        if self.is_ready:
            return False  # stop timer
        elapsed = int(time.time() - self.load_start)
        remaining = max(0, LOAD_SECONDS - elapsed)
        self.lbl_countdown.set_text(f"~{remaining}s")
        return True  # keep going

    # ── called from bg thread when model ready ────────
    def set_ready(self):
        GLib.idle_add(self._go_ready)

    def _go_ready(self):
        self.is_ready = True
        self.lbl_title.set_text(f"{self.icon_str} Listening...")
        self.lbl_countdown.set_text("")
        self.lbl_hint.set_text("")
        self.lbl_status.set_text("Speak freely — close window to stop & save")
        self.btn.set_sensitive(True)

    def set_saving(self):
        self.is_saving = True
        self.lbl_title.set_text("⏳ Saving...")
        self.lbl_hint.set_text("Processing remaining audio and saving...")
        self.lbl_status.set_text("Please wait or Force Quit")
        self.btn.set_label("Force Quit")
        try:
            self.btn.disconnect_by_func(self._on_close)
        except Exception:
            pass
        self.btn.connect("clicked", self._force_quit)
        self.btn.set_sensitive(True)

    def set_processing(self):
        self.lbl_title.set_text("⏳ Processing with AI...")
        self.lbl_hint.set_text("Gemini is polishing your transcription...")

    def set_syncing(self):
        self.lbl_title.set_text("⏳ Syncing...")
        self.lbl_hint.set_text("Saving to Obsidian and syncing to Git...")

    def _force_quit(self, *args):
        try:
            _release_lock()
        except Exception:
            pass
        os._exit(1)

    def increment_chunk(self):
        self.chunks += 1
        GLib.idle_add(self._update_chunks)

    def _update_chunks(self):
        n = self.chunks
        self.lbl_chunks.set_text(
            f"{n} chunk{'s' if n != 1 else ''} transcribed")

    # ── animation tick ────────────────────────────────
    def _tick(self):
        self.phase += 0.18 if self.is_ready else 0.06
        self.canvas.queue_draw()
        return True

    def _draw(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        cx, cy = w / 2, h / 2

        if not self.is_ready:
            # pulsing ring countdown
            elapsed  = time.time() - self.load_start
            progress = min(elapsed / LOAD_SECONDS, 1.0)
            radius   = 28
            lw       = 5

            # background ring
            cr.set_source_rgba(0.19, 0.19, 0.30, 1)
            cr.set_line_width(lw)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.stroke()

            # progress arc (red → green as it fills)
            r = 0.95 - 0.55 * progress
            g = 0.27 + 0.62 * progress
            b = 0.42
            cr.set_source_rgba(r, g, b, 1)
            cr.set_line_width(lw)
            start = -math.pi / 2
            end   = start + 2 * math.pi * progress
            cr.arc(cx, cy, radius, start, end)
            cr.stroke()

            # soft pulse dot in center
            pulse = 4 + 2 * math.sin(self.phase)
            cr.set_source_rgba(r, g, b, 0.4)
            cr.arc(cx, cy, pulse, 0, 2 * math.pi)
            cr.fill()
            return

        if self.is_saving:
            radius = 28
            lw     = 5
            cr.set_source_rgba(0.19, 0.19, 0.30, 1)
            cr.set_line_width(lw)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.stroke()

            cr.set_source_rgba(0.65, 0.89, 0.63, 1)
            cr.set_line_width(lw)
            start = self.phase * 2.5
            end   = start + math.pi / 2
            cr.arc(cx, cy, radius, start, end)
            cr.stroke()
            return

        # ── Live microphone-reactive waveform ────────────────────────
        bars  = 20
        bw    = 8
        gap   = 5
        total = bars * (bw + gap) - gap
        x0    = cx - total / 2

        # audio_level: 0.0 = silence, 1.0 = loud voice (written by _level_monitor thread)
        level = self.audio_level

        for i in range(bars):
            # Gentle lateral shimmer: phase-driven, but weight fades when speaking
            shimmer = abs(math.sin(self.phase + i * 0.42))

            # Bar height:
            #   silent  (level≈0) → 3–7 px idle shimmer
            #   loud    (level≈1) → 8–55 px reactive peak
            bar_h = 3 + shimmer * 4 * (1.0 - level) + level * (8 + shimmer * 47)

            x     = x0 + i * (bw + gap)
            y     = cy - bar_h / 2

            # Alpha: dim at idle, bright when speaking
            alpha = 0.25 + 0.15 * shimmer * (1.0 - level) + level * (0.65 + 0.35 * shimmer)

            # Color: soft purple at idle → bright violet when loud
            r = 0.80 + level * 0.15
            g = 0.65 + level * 0.20
            b = 0.97
            cr.set_source_rgba(r, g, b, alpha)
            self._rrect(cr, x, y, bw, bar_h, 3)
            cr.fill()

    def _rrect(self, cr, x, y, w, h, r):
        cr.move_to(x + r, y)
        cr.line_to(x + w - r, y)
        cr.arc(x + w - r, y + r, r, -math.pi/2, 0)
        cr.line_to(x + w, y + h - r)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi/2)
        cr.line_to(x + r, y + h)
        cr.arc(x + r, y + h - r, r, math.pi/2, math.pi)
        cr.line_to(x, y + r)
        cr.arc(x + r, y + r, r, math.pi, 3*math.pi/2)
        cr.close_path()

    def _on_close(self, *args):
        self.on_stop()
        return True  # Prevent window close until fully processed


# ── live microphone level monitor ────────────────────
def _level_monitor(win_ref, stop_event):
    """
    Daemon thread: opens a lightweight PyAudio input stream, reads audio
    every ~64 ms, computes RMS amplitude, applies asymmetric exponential
    smoothing, and writes the result into win_ref[0].audio_level (0.0–1.0).

    Uses a completely separate stream from the main record() thread so it
    never steals frames from the Whisper audio queue.
    """
    CHUNK   = 1024    # ~64 ms at 16 kHz
    RATE    = 16000
    SMOOTH  = 0.25    # attack: how fast level rises  (higher = snappier)
    DECAY   = 0.125   # release: how fast level falls  (lower  = slower decay)
    SILENCE = 0.005   # RMS below this threshold → treat as silence
    PEAK    = 0.15    # RMS that maps to audio_level = 1.0

    try:
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        smoothed = 0.0

        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                break

            samples   = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms       = float(np.sqrt(np.mean(samples ** 2)))
            raw_level = min(rms / PEAK, 1.0) if rms > SILENCE else 0.0

            # Asymmetric smoothing: fast attack, slow decay
            if raw_level > smoothed:
                smoothed = SMOOTH * raw_level + (1.0 - SMOOTH) * smoothed
            else:
                smoothed = DECAY  * raw_level + (1.0 - DECAY)  * smoothed

            # GIL makes float assignment atomic — no lock needed
            if win_ref[0] is not None:
                win_ref[0].audio_level = smoothed

        stream.stop_stream()
        stream.close()
        pa.terminate()
    except Exception:
        logger.warning("Level monitor failed to start — waveform will use idle shimmer",
                       exc_info=True)


# ── core runner ───────────────────────────────────────
def run(mode):
    # ── enforce single instance ───────────────────────
    # Kill any existing whisper session so only one runs at a time.
    # The old session's SIGTERM handler commits its data before exiting.
    _kill_existing()
    _write_lock()
    if mode != "system":
        _check_recovery_file()
    stop_record_event = threading.Event()
    audio_queue       = queue.Queue()

    # out_path is a list so inner functions can mutate it (closure reference)
    out_path = [None]

    # ── system mode: remember the currently focused window BEFORE our GTK
    #    overlay steals the screen, so we can focus it back before typing.
    target_wid = [None]
    if mode == "system":
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=3
            )
            wid = result.stdout.strip()
            if wid.isdigit():
                target_wid[0] = wid
        except Exception:
            pass  # xdotool unavailable — will still attempt to type

    if mode == "journal":
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        out_path[0] = os.path.join(JOURNAL_DIR, datetime.now().strftime("%Y-%m-%d") + ".md")
        # Create from template if this daily note doesn't exist yet
        if not os.path.exists(out_path[0]):
            if os.path.exists(JOURNAL_TPL):
                with open(JOURNAL_TPL, "r", encoding="utf-8") as tpl:
                    tpl_content = tpl.read()
                # Substitute Templater placeholders with real values
                today = datetime.now()
                tpl_content = re.sub(
                    r"<% tp\.date\.now\(['\"]YYYY-MM-DD['\"]\) %>",
                    today.strftime("%Y-%m-%d"), tpl_content)
                tpl_content = re.sub(
                    r"<% tp\.date\.now\(['\"]dddd, MMMM DD, YYYY['\"]\) %>",
                    today.strftime("%A, %B %d, %Y"), tpl_content)
                with open(out_path[0], "w", encoding="utf-8") as f:
                    f.write(tpl_content)
            else:
                # Fallback: minimal daily note with Voice Capture section
                today = datetime.now()
                with open(out_path[0], "w", encoding="utf-8") as f:
                    f.write(
                        f"---\ndate: {today.strftime('%Y-%m-%d')}\ntype: daily\n"
                        f"focus_score: \nenergy_score: \nexecution_score: \n"
                        f"did_high_priority: \ndid_income_work: \ndid_learning: \n---\n\n"
                        f"# 📅 {today.strftime('%A, %B %d, %Y')}\n\n"
                        f"## 🔥 Highest Priority\n- [ ] \n\n---\n\n"
                        f"## 💰 Income Work\n- [ ] \n\n---\n\n"
                        f"## 🧠 Learning\n- [ ] \n\n---\n\n"
                        f"## ✅ Tasks\n- [ ] \n\n---\n\n"
                        f"## 🏆 One Win\n\n---\n\n"
                        f"## ❌ One Thing to Fix\n\n---\n\n"
                        f"## 📊 Score\nFocus: /10  \nEnergy: /10  \nExecution: /10  \n\n---\n\n"
                        f"## 📝 How Today Went\n\n---\n\n"
                        f"## 🎙 Voice Capture\n\n---\n\n"
                        f"## ✅ Daily Summary\n- Highest priority done: \n- Income work done: \n- Learning done: \n"
                    )
    elif mode == "inbox":
        os.makedirs(INBOX_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-idea-%H-%M")
        out_path[0] = os.path.join(INBOX_DIR, f"{ts}.md")
        with open(out_path[0], "w") as f:
            f.write(f"# Idea — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    elif mode == "book":
        os.makedirs(BOOKS_DIR, exist_ok=True)
        # For 'book' mode, the path will dynamically resolve in transcribe() based on the first few words.
        pass

    win        = [None]
    level_stop = threading.Event()   # signals _level_monitor to exit

    def on_stop():
        """Called when the user closes the window / clicks Stop."""
        if stop_record_event.is_set():
            return
        stop_record_event.set()
        level_stop.set()              # stop the level-monitor thread
        if win[0]:
            GLib.idle_add(win[0].set_saving)

    def build_window():
        win[0] = WhisperWindow(mode, on_stop)

        # ── GTK-safe SIGTERM handler ──────────────────
        # When a NEW shortcut is pressed, _kill_existing() sends SIGTERM here.
        # GLib.unix_signal_add handles it safely inside the main loop.
        def _on_sigterm():
            on_stop()
            return GLib.SOURCE_REMOVE

        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, _on_sigterm)
        # ─────────────────────────────────────────────

        # Start the live microphone level monitor for the reactive waveform.
        # Runs only after the window exists so win_ref[0] is already set.
        threading.Thread(
            target=_level_monitor,
            args=(win, level_stop),
            daemon=True,
            name="LevelMonitor",
        ).start()

    GLib.idle_add(build_window)

    def load_and_run():
        try:
            logger.info("Loading WhisperModel...")
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            sys.stderr = sys.__stderr__
            logger.info("Model loaded successfully")

            if win[0]:
                win[0].set_ready()
        except Exception as e:
            err_msg = str(e)
            logger.error("Model load error", exc_info=True)
            GLib.idle_add(lambda msg=err_msg: notify("Whisper Error", f"Failed to load model: {msg}"))
            GLib.idle_add(lambda: Gtk.main_quit())
            return

        def record():
            try:
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1,
                                 rate=SAMPLE_RATE, input=True,
                                 frames_per_buffer=1024)
                needed = int(SAMPLE_RATE * CHUNK_SECS)
                while not stop_record_event.is_set():
                    frames = []
                    for _ in range(needed // 1024):
                        if stop_record_event.is_set():
                            break
                        data = stream.read(1024, exception_on_overflow=False)
                        frames.append(np.frombuffer(data, dtype=np.int16))
                    if frames:
                        audio_queue.put(np.concatenate(frames))
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except Exception as e:
                logger.error("Recording error", exc_info=True)
                on_stop()

        def transcribe():
            session_text = []
            try:
                if mode != "system":
                    open(RECOVERY_FILE, "w", encoding="utf-8").close()

                while True:
                    try:
                        chunk = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_record_event.is_set():
                            break
                        continue
                    audio = chunk.astype(np.float32) / 32768.0
                    if np.abs(audio).mean() < 0.005:
                        continue
                    segments, _ = model.transcribe(
                        audio, language=LANGUAGE,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=True,
                        initial_prompt=INITIAL_PROMPT,
                        temperature=0.0,               # deterministic output — kills punctuation randomness
                        compression_ratio_threshold=2.4, # reject repetitive/looping output
                        log_prob_threshold=-1.0,        # skip very uncertain segments
                        no_speech_threshold=0.6,        # stronger silence filter
                    )
                    segments_list = list(segments)
                    raw_text  = _smart_join(segments_list).strip()
                    corrected = _apply_corrections(raw_text)

                    if mode == "system":
                        text = _gemini_polish(corrected)
                        if not text:
                            continue
                        cmd = ["xdotool"]
                        if target_wid[0]:
                            cmd += ["windowfocus", "--sync", target_wid[0],
                                    "type", "--window", target_wid[0],
                                    "--clearmodifiers", "--delay", "15", text + " "]
                        else:
                            cmd += ["type", "--clearmodifiers", "--delay", "15", text + " "]
                        subprocess.Popen(cmd)
                    else:
                        if not corrected:
                            continue
                        session_text.append(corrected)
                        with open(RECOVERY_FILE, "a", encoding="utf-8") as rf:
                            rf.write(corrected + "\n\n")
                            rf.flush()
                        if mode == "book" and out_path[0] is None and len(session_text) >= 2:
                            full_accumulated = "\n\n".join(session_text)
                            best_book, confidence = _match_book(full_accumulated, BOOKS_DIR)
                            if confidence > 0.15 and best_book:
                                out_path[0] = os.path.join(BOOKS_DIR, best_book)
                                logger.info(f"Book matched: '{best_book}' (confidence={confidence:.2f})")
                            else:
                                out_path[0] = os.path.join(BOOKS_DIR, "Unsorted_Notes.md")
                                logger.warning(f"No book matched (best={best_book}, conf={confidence:.2f}). Saving to Unsorted.")

                    if win[0]:
                        win[0].increment_chunk()
            except Exception as e:
                logger.error("Transcription error", exc_info=True)
            finally:
                # --- TRANSCRIPTION FINISHED, CLEANUP AND EXIT ---
                def do_save_bg():
                    _release_lock()
                    if mode == "system":
                        notify("Whisper", "✅ Dictation stopped.")
                    else:
                        if session_text:
                            if mode == "book" and out_path[0] is None:
                                out_path[0] = os.path.join(BOOKS_DIR, "Unsorted_Notes.md")
                            full_text = "\n\n".join(session_text)
                            if win[0]:
                                GLib.idle_add(win[0].set_processing)
                            polished = _gemini_polish_safe(full_text)
                            if win[0]:
                                GLib.idle_add(win[0].set_syncing)
                            if mode == "journal":
                                _insert_into_voice_capture(out_path[0], polished)
                            elif mode == "inbox":
                                with open(out_path[0], "a", encoding="utf-8") as f:
                                    f.write(f"\n{polished}")
                            elif mode == "book":
                                ts = datetime.now().strftime('%Y-%m-%d %H:%M')
                                with open(out_path[0], "a", encoding="utf-8") as f:
                                    if "Unsorted_Notes" in out_path[0]:
                                        f.write(f"\n\n# Unsorted Note — {ts}\n{polished}")
                                    else:
                                        f.write(f"\n\n**[{ts}] Insight:** {polished}")
                            try:
                                os.remove(RECOVERY_FILE)
                            except OSError:
                                pass
                        if out_path[0]:
                            committed = _git_commit_sync(mode, out_path[0])
                            if committed:
                                try:
                                    subprocess.run(["git", "-C", VAULT_PATH, "push"], capture_output=True, timeout=60, check=True)
                                    if mode == "journal":
                                        label = "✅ Journal saved & pushed"
                                    elif mode == "book":
                                        label = "✅ Book insight saved & pushed"
                                    else:
                                        label = "✅ Idea saved & pushed"
                                except Exception as e:
                                    label = f"⚠️ Commit saved, push failed: {e}"
                            else:
                                label = "⏹ Nothing to commit"
                            notify("Whisper", label)
                    
                    logger.info(f"Session finished (mode: {mode})")
                    # Close UI cleanly on main thread
                    def _close_ui():
                        if win[0]:
                            win[0].destroy()
                        Gtk.main_quit()
                        return False
                    
                    GLib.idle_add(_close_ui)

                threading.Thread(target=do_save_bg, daemon=False).start()

        threading.Thread(target=record, daemon=True).start()
        threading.Thread(target=transcribe, daemon=True).start()

    threading.Thread(target=load_and_run, daemon=True).start()
    Gtk.main()
