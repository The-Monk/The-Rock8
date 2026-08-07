#!/usr/bin/env bash
# fetch-model.sh — download a Rock8 fp8 GGUF from Hugging Face.
#
# Filenames are NOT uniform across these repos (some say Quark, one says
# Quacken), which is exactly the sort of detail that breaks a copy-pasted curl.
# This resolves the real filename from the HF API rather than assuming it.
#
#   ./fetch-model.sh                 # list what is available
#   ./fetch-model.sh 8b              # download to ./models
#   MODELS_DIR=/data ./fetch-model.sh 27b
set -uo pipefail
DEST="${MODELS_DIR:-./models}"

declare -A REPO=(
  [8b]=Quacken-8B-FP8
  [14b]=Quacken-R1-14B-FP8
  [27b]=Quacken-27B-FP8
  [35b]=Quacken-35B-A3B-FP8
  [ornith]=Quacken-Ornith-35B-FP8
)
declare -A NOTE=(
  [8b]="Qwen3-8B — smallest, best first test"
  [14b]="DeepSeek-R1-Distill-Qwen-14B — reasoning"
  [27b]="Qwen3.6-27B dense"
  [35b]="Qwen3.6-35B-A3B MoE — needs ~40 GB"
  [ornith]="Ornith-1.0-35B"
)

if [ $# -eq 0 ]; then
  echo "Rock8 fp8 models (each is one GGUF; sizes are approximate):"
  for k in 8b 14b 27b 35b ornith; do printf "  %-8s %s\n" "$k" "${NOTE[$k]}"; done
  echo
  echo "usage: $0 <key>        # downloads into ${DEST}"
  echo "Collection: https://huggingface.co/collections/Gorilla4X/the-rock8-rdna4-fp8-6a547070f667cb41db0bc2ed"
  echo
  echo "Licenses follow the SOURCE model (Apache-2.0 / MIT); see each model card."
  exit 0
fi

KEY="${1,,}"
[ -n "${REPO[$KEY]:-}" ] || { echo "unknown model '$KEY' -- run with no arguments to list" >&2; exit 1; }
R="${REPO[$KEY]}"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
echo "resolving filename for Gorilla4X/$R ..."
FILE=$(curl -fsSL "https://huggingface.co/api/models/Gorilla4X/$R" \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
g=[x['rfilename'] for x in d.get('siblings',[]) if x['rfilename'].endswith('.gguf')]
print(g[0] if g else '')" 2>/dev/null)
[ -n "$FILE" ] || { echo "could not resolve a .gguf in that repo -- check the collection link" >&2; exit 1; }

mkdir -p "$DEST"
echo "downloading $FILE -> $DEST/"
curl -fL --progress-bar -o "$DEST/$FILE" \
  "https://huggingface.co/Gorilla4X/$R/resolve/main/$FILE"

echo
echo "done: $DEST/$FILE"
echo "run it:"
echo "  podman run --rm --runtime crun --device /dev/kfd --device /dev/dri \\"
echo "    --group-add keep-groups --security-opt seccomp=unconfined \\"
echo "    -v \"\$(cd \"$DEST\" && pwd)\":/models:ro -e MODEL=/models/$FILE \\"
echo "    ghcr.io/the-monk/the-rock8:rdna4-tr713 bench"
