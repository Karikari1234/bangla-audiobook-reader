#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: two Python envs, one per TTS engine.
#   .venv       app server + Kokoro (English, PyTorch on MPS)
#   .venv-omni  OmniVoice-MLX (Bangla, MLX on the Apple GPU)
set -euo pipefail
cd "$(dirname "$0")"

[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "Needs an Apple Silicon Mac (M1 or newer)."; exit 1; }
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
command -v uv >/dev/null || brew install uv
brew list espeak-ng >/dev/null 2>&1 || brew install espeak-ng   # Kokoro's English phonemizer

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

uv venv --python 3.12 .venv-omni
uv pip install --python .venv-omni/bin/python "omnivoice[mlx] @ git+https://github.com/Karikari1234/OmniVoice-MLX@speedup-bangla"

# Fetch the Bangla model now (~1.4 GB) so the first paragraph doesn't wait on a download.
.venv-omni/bin/python -c "from huggingface_hub import snapshot_download as s; s('mlx-community/OmniVoice-8bit')"

echo
echo "Done. Start it with:  ./run.sh ~/Books   (any folder of PDF/EPUB/TXT books)"
