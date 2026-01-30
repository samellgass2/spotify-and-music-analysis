#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/spotify-and-music-analysis/Scripts"

# Load secrets (client id/secret + refresh tokens)
source ./secrets.env

# Activate venv
source ./.venv/bin/activate

# Run (weekday override happens inside the script)
python3 ./daily-drive-but-better.py run --profile "${1:-sam}" ${2:-} 
