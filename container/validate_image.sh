#!/usr/bin/env bash
# Validate the BUILT image (no therock/llama bind-mounts — only /models).
set -uo pipefail
IMG=roc8-lemonade:tr713
MODELS=/aipool/models/qwen3-8b-fp8
SC=/tmp/claude-1000/-home-jmonk/5f7e73a6-0432-4b0d-b7aa-bae8db6133e2/scratchpad
RUN="podman run --rm --runtime crun --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined -v $MODELS:/models:ro"

echo "########## A) ldd closure from built image (zero /opt/rocm) ##########"
$RUN --entrypoint bash $IMG -c '
  for b in llama-bench llama-perplexity llama-server llama-quantize; do
    echo "### $b ###"; ldd /opt/llama/$b 2>&1 | grep -iE "not found|/opt/rocm" || echo "  full closure, no /opt/rocm";
  done
  echo "### rocm sonames source ###"; ldd /opt/llama/libggml-hip.so.0 | grep -iE "rocblas|hipblas|amdhip" ' 2>&1 | tee $SC/img_ldd.txt

echo "########## B) llama-bench from built image ##########"
$RUN --entrypoint /opt/llama/llama-bench $IMG \
  -m /models/Qwen3-8B-F8E4M3.gguf -p 128 -n 32 -ngl 99 2>&1 | grep -iE "Device 0|qwen3|error" | tee $SC/img_bench.txt

echo "########## C) llama-perplexity from built image ##########"
$RUN -v /aipool/models/qwen3.6-27b-bf16-mtp/test-run/wikitext/wikitext-2-raw:/wiki:ro \
  --entrypoint /opt/llama/llama-perplexity $IMG \
  -m /models/Qwen3-8B-F8E4M3.gguf -f /wiki/wiki.test.raw --chunks 20 -ngl 99 2>&1 | grep -iE "Final estimate" | tee $SC/img_ppl.txt

echo "DONE"
