#!/usr/bin/env bash
# Set up the demo on first run, then start it at http://127.0.0.1:7860
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Kept outside the project so Google Drive does not sync the environment.
VENV="${CMR_DEMO_VENV:-$HOME/.cache/cmr-scar-demo-venv}"

if [ ! -f "$VENV/.installed" ] || [ "$DEMO_DIR/requirements.txt" -nt "$VENV/.installed" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --disable-pip-version-check -r "$DEMO_DIR/requirements.txt"
    touch "$VENV/.installed"
fi

exec "$VENV/bin/python" "$DEMO_DIR/app.py"
