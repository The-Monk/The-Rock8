#!/usr/bin/env bash
# Entry for the ROC8 + Lemonade / TheRock 7.13 appliance.
#   serve   -> register the fp8 model (if a GGUF is mounted) and start lemonade-server
#   bench   -> run llama-bench on the mounted model
#   ppl     -> run llama-perplexity on the mounted model + wikitext
#   bash    -> drop to a shell
set -euo pipefail

MODEL="${MODEL:-/models/Qwen3-8B-F8E4M3.gguf}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B-FP8}"
LEM_CACHE="${LEMONADE_CACHE_DIR:-/root/.cache/lemonade}"
PORT="${LEMONADE_PORT:-13305}"

register_model() {
  # Direct-path checkpoint: lemonade uses it verbatim when the file exists.
  mkdir -p "$LEM_CACHE"
  /opt/venv/bin/python - "$MODEL_NAME" "$MODEL" "$LEM_CACHE" <<'PY'
import json, os, sys
name, ckpt, cache = sys.argv[1], sys.argv[2], sys.argv[3]
f = os.path.join(cache, "user_models.json")
data = {}
if os.path.exists(f):
    data = json.load(open(f))
data[name] = {"checkpoint": ckpt, "recipe": "llamacpp",
              "suggested": True, "labels": ["custom"], "source": "local_upload"}
json.dump(data, open(f, "w"))
print(f"registered {name} -> {ckpt}")
PY
  # Force rocm backend + our binaries
  CFG="$LEM_CACHE/config.json"
  /opt/venv/bin/python - "$CFG" <<'PY'
import json, os, sys
f = sys.argv[1]
c = json.load(open(f)) if os.path.exists(f) else {}
c.setdefault("llamacpp", {})
c["llamacpp"]["backend"] = "rocm"
c["llamacpp"]["prefer_system"] = False
c["port"] = int(os.environ.get("LEMONADE_PORT", "13305"))
c["host"] = "0.0.0.0"
json.dump(c, open(f, "w"))
print("config wired: backend=rocm host=0.0.0.0")
PY
}

case "${1:-serve}" in
  serve)
    [ -f "$MODEL" ] && register_model || echo "WARN: $MODEL not mounted; starting lemonade without preregistered model"
    exec /opt/venv/bin/lemonade-server-dev serve --host 0.0.0.0 --port "$PORT" --llamacpp rocm
    ;;
  bench)
    shift
    exec /opt/llama/llama-bench -m "$MODEL" "${@:--p 128 -n 32 -ngl 99}"
    ;;
  ppl)
    shift
    exec /opt/llama/llama-perplexity -m "$MODEL" "$@"
    ;;
  bash|sh)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
