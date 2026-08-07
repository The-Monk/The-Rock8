#!/usr/bin/env bash
# Validate the BUILT image (no therock/llama bind-mounts — only /models).
set -uo pipefail
IMG="${IMG:-ghcr.io/the-monk/the-rock8:rdna4-tr713}"
MODELS=${MODELS:?set MODELS to the directory holding your GGUFs}
SC="${SCRATCH:-./scratch}"
mkdir -p "$SC"
: "${MODELS:?set MODELS to the directory holding your GGUFs}"
: "${WIKITEXT}"
RUN="podman run --rm --runtime crun --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined -v $MODELS:/models:ro"

echo "########## A) ldd closure from built image (zero /opt/rocm) ##########"
$RUN --entrypoint bash $IMG -c '
  # llama-completion is in this list deliberately: upstream b9963 made llama-cli
  # conversation-only and moved one-shot completion here. An image without it has
  # no scriptable generation path at all, and the failure surfaces to the user as
  # a hang rather than an error, so its absence must fail the build.
  for b in llama-bench llama-perplexity llama-server llama-quantize llama-cli llama-completion; do
    echo "### $b ###";
    test -x /opt/llama/$b || { echo "  FATAL: /opt/llama/$b missing or not executable"; exit 1; };
    ldd /opt/llama/$b 2>&1 | grep -iE "not found|/opt/rocm" || echo "  full closure, no /opt/rocm";
  done
  echo "### rocm sonames source ###"; ldd /opt/llama/libggml-hip.so.0 | grep -iE "rocblas|hipblas|amdhip" ' 2>&1 | tee $SC/img_ldd.txt

echo "########## B) llama-bench from built image ##########"
$RUN --entrypoint /opt/llama/llama-bench $IMG \
  -m /models/Qwen3-8B-Quark-F8E4M3.gguf -p 128 -n 32 -ngl 99 2>&1 | grep -iE "Device 0|qwen3|error" | tee $SC/img_bench.txt

echo "########## C) llama-perplexity from built image ##########"
$RUN -v ${WIKITEXT}:/wiki:ro \
  --entrypoint /opt/llama/llama-perplexity $IMG \
  -m /models/Qwen3-8B-Quark-F8E4M3.gguf -f /wiki/wiki.test.raw --chunks 20 -ngl 99 2>&1 | grep -iE "Final estimate" | tee $SC/img_ppl.txt

echo "DONE"
