#!/bin/bash
# ── Whisper Setup Script ───────────────────────────────
# Run once: bash setup.sh
# ──────────────────────────────────────────────────────

SCRIPTS_DIR="$HOME/Courses/Notes/fast-whisper-to-obsidian-sync"
VENV_PYTHON="$HOME/whisper-env/bin/python3"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Whisper Obsidian Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "▸ Installing system dependencies..."
sudo apt install -y xdotool libnotify-bin python3-gi python3-gi-cairo gir1.2-gtk-3.0

echo ""
echo "▸ Installing PyGObject in venv..."
source "$HOME/whisper-env/bin/activate"
pip install PyGObject 2>/dev/null || true

echo ""
echo "▸ Setting permissions..."
chmod +x "$SCRIPTS_DIR/whisper_base.py"
chmod +x "$SCRIPTS_DIR/transcribe_journal.py"
chmod +x "$SCRIPTS_DIR/transcribe_inbox.py"
chmod +x "$SCRIPTS_DIR/transcribe_system.py"

echo ""
echo "▸ Registering keyboard shortcuts..."

SCHEMA="org.gnome.settings-daemon.plugins.media-keys"
BASE="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"

gsettings set $SCHEMA custom-keybindings \
  "['$BASE/custom0/', '$BASE/custom1/', '$BASE/custom2/', '$BASE/custom3/']"

gsettings set $SCHEMA.custom-keybinding:$BASE/custom0/ name    "Whisper Journal"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom0/ command "$VENV_PYTHON $SCRIPTS_DIR/transcribe_journal.py"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom0/ binding "<Primary><Shift>j"

gsettings set $SCHEMA.custom-keybinding:$BASE/custom1/ name    "Whisper Inbox"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom1/ command "$VENV_PYTHON $SCRIPTS_DIR/transcribe_inbox.py"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom1/ binding "<Primary><Shift>i"

gsettings set $SCHEMA.custom-keybinding:$BASE/custom2/ name    "Whisper Dictate"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom2/ command "$VENV_PYTHON $SCRIPTS_DIR/transcribe_system.py"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom2/ binding "<Primary><Shift>s"

gsettings set $SCHEMA.custom-keybinding:$BASE/custom3/ name    "Whisper Book"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom3/ command "$VENV_PYTHON $SCRIPTS_DIR/transcribe_book.py"
gsettings set $SCHEMA.custom-keybinding:$BASE/custom3/ binding "<Primary><Shift>b"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Ctrl+Shift+J  →  Journal"
echo "  Ctrl+Shift+I  →  Inbox"
echo "  Ctrl+Shift+B  →  Book Insights"
echo "  Ctrl+Shift+S  →  System dictation"
echo ""
echo "  Press shortcut → wait ~40s → speak"
echo "  Close window → saved + pushed to GitHub"
echo "  RAM freed completely when window closes."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
