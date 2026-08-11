#!/usr/bin/env bash
# Lemonade round-trip from the BUILT image. Proves completion goes THROUGH lemonade.
set -uo pipefail
IMG="${IMG:-ghcr.io/the-monk/the-rock8:rdna4-tr713}"
MODELS=${MODELS:?set MODELS to the directory holding your GGUFs}
# MODEL must name the GGUF explicitly: the image's baked-in default may not
# match the file you mounted, and a mismatch means lemonade starts with no
# model registered and every chat request 422s (caught by the 2026-08-10
# validation pass, where this script could not pass against the published
# artifacts for exactly that reason).
MODEL="${MODEL:-/models/Qwen3-8B-Quark-F8E4M3.gguf}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B-FP8}"
PORT=13405   # host port (avoid clashing with any host lemond on 13305)
NAME=roc8lem
SC=${SCRATCH:-./scratch}

podman rm -f $NAME >/dev/null 2>&1

echo "=== starting lemonade container (detached) ==="
podman run -d --rm --runtime crun --name $NAME \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v $MODELS:/models:ro \
  -e MODEL="$MODEL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e LEMONADE_PORT=13305 \
  -p ${PORT}:13305 \
  $IMG serve
echo "container id started; waiting for lemonade health..."

# Poll health up to ~120s
ok=0
for i in $(seq 1 60); do
  h=$(curl -s -m 3 http://localhost:${PORT}/api/v1/health 2>/dev/null)
  if echo "$h" | grep -qiE 'ok|status|true|healthy|model'; then echo "HEALTH[$i]: $h"; ok=1; break; fi
  sleep 2
done
[ $ok -eq 0 ] && { echo "HEALTH NEVER CAME UP; container logs:"; podman logs $NAME 2>&1 | tail -40; }

echo "=== models list ==="
curl -s -m 5 http://localhost:${PORT}/api/v1/models 2>/dev/null | head -c 600; echo

echo "=== chat completion THROUGH lemonade ==="
RESP=$(curl -s -m 180 http://localhost:${PORT}/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"user.${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What do you call a dried grape? Answer in one word. /no_think\"}],\"max_tokens\":32,\"temperature\":0}")
echo "$RESP" | tee $SC/lemonade_resp.json | head -c 1200; echo
echo "=== extracted content ==="
echo "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' 2>/dev/null || echo "(parse failed)"

echo "=== container log tail ==="
podman logs $NAME 2>&1 | tail -25
echo "=== stopping ==="
podman stop $NAME >/dev/null 2>&1
echo "LEMONADE_TEST_DONE"
