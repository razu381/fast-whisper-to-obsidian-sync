#!/usr/bin/env bash
# setup_gemini_key.sh
# One-time helper: saves your Gemini API key to ~/.bashrc so whisper_base.py
# can pick it up via os.environ.get("GEMINI_API_KEY", "")
#
# Usage:  bash setup_gemini_key.sh YOUR_API_KEY_HERE
#         Get a free key at: https://aistudio.google.com/app/apikey

set -e

KEY="$1"

if [ -z "$KEY" ]; then
    echo "Usage: bash setup_gemini_key.sh YOUR_GEMINI_API_KEY"
    echo "Get a free key at: https://aistudio.google.com/app/apikey"
    exit 1
fi

# Remove any existing GEMINI_API_KEY line from ~/.bashrc to avoid duplicates
sed -i '/^export GEMINI_API_KEY=/d' ~/.bashrc

# Append the new key
echo "export GEMINI_API_KEY=\"${KEY}\"" >> ~/.bashrc

# Also export it into the current shell session immediately
export GEMINI_API_KEY="${KEY}"

echo ""
echo "✅ GEMINI_API_KEY saved to ~/.bashrc"
echo "   Key prefix: ${KEY:0:12}..."
echo ""
echo "Reload your shell or run: source ~/.bashrc"
echo "Then test with:  python3 -c \"import google.generativeai as g; g.configure(api_key='${KEY}'); print(g.GenerativeModel('gemini-2.0-flash').generate_content('say hi').text)\""
