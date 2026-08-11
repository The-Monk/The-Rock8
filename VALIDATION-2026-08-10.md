# Full validation pass — 2026-08-10

Re-scan of every shippable claim in this repository, executed on real hardware.
Everything below was run on 2026-08-10 against a fresh `git clone` of this repo
at `28773c8` (master HEAD) — not a long-lived working tree.

**Environment:** Ubuntu 24.04, rootless Podman + crun, 2× AMD Radeon AI PRO
R9700 (gfx1201). Published image under test:
`ghcr.io/the-monk/the-rock8:rdna4-tr713`, image id `ad8b65881b84`, build
`f36df0f3c (252)`.

## Verdict

The stack itself is solid — GPU inference, serving, published models and the
recursive clone all check out end-to-end. **One thing blocks a clean ship:
the README §0 quickstart `bench` command fails against the published image**
(entrypoint quoting bug, fixed in this repo but never rebuilt into the image).
Rebuild and push the image — after fixing the `LLAMA_ARGS` regression noted
below — or amend §0 to pass the bench args explicitly.

## What passed

| Check | Result |
|---|---|
| `git clone --recursive` | submodule pin `3114c471f` == public `roc8` HEAD; checks out clean |
| HF model repos | all 5 public; GGUF filenames exactly as the README lists them |
| `./fetch-model.sh` (list mode) | correct listing, exit 0 |
| `./fetch-model.sh 8b` | resolved `Qwen3-8B-Quark-F8E4M3.gguf` via HF API, downloaded 9.2 GB clean |
| Published artifact integrity | sha256 `8b3ef9399b99…3ccb056b` — HF download byte-identical to the local staging copy |
| ghcr anonymous pull | token + manifest fetch = HTTP 200, no auth |
| GPU visibility | `Device 0: AMD Radeon AI PRO R9700, gfx1201` inside the container |
| `llama-bench` (explicit args) | pp128 **2971–3004 t/s**, tg32 **59.6–59.7 t/s** (two runs, 1× R9700) |
| `validate_image.sh` A) ldd closure | all binaries fully closed, zero `/opt/rocm` — except the missing `llama-completion` (below) |
| `validate_image.sh` C) perplexity | `PPL = 10.9345 ± 0.48054` (wikitext-2, 20 chunks) — the README's reproducibility claim holds |
| README §4 serve round-trip | model auto-registered, chat through Lemonade answered correctly; 439 t/s prompt / 63.2 t/s gen; fingerprint `b252-f36df0f3c` |
| Cross-repo links | `toolkit/rdna4-memprof.sh`, `toolkit/rdna4-fetchsize-repro/`, `kernels/decode/agent_repro/`, `benchmarks/crossover.sh` all resolve in rocky-hackathon; ROCm/composable_kernel#3759 open with matching title |

## Findings (ship-relevant, most severe first)

1. **README §0 quickstart is broken against the published image.** The image's
   entrypoint runs `"${@:--p 128 -n 32 -ngl 99}"`, which passes the default
   bench args as a *single* word; llama-bench prints its usage and exits after
   the `Device 0` line, so a new user following §0 verbatim never gets a
   tokens/sec figure. This repo's `container/entrypoint.sh` already fixes the
   quoting — the fix just isn't in any pushed image. Workaround that works
   today: `… bench -p 128 -n 32 -ngl 99` (verified above).

2. **`/opt/llama/llama-completion` is absent from the published image.**
   `libllama-completion-impl.so` is present; the wrapper binary is not.
   `validate_image.sh` FATALs on it by design ("its absence must fail the
   build"), and the README §4 one-shot `llama-completion` example cannot work
   in the image you pull. A rebuild picks it up.

3. **Regression in this repo's `container/entrypoint.sh`:** it sets
   `LLAMA_ARGS="${LLAMA_ARGS:--ngl 999 --cont-batching}"` — with a comment
   saying the args "have to actually be passed" — but the variable is never
   exported and never referenced again. The `recipe_options.llamacpp_args`
   wiring that the *published image's* entrypoint has was dropped, so an image
   rebuilt from this tree would register the model with **no** llamacpp args:
   no `--cont-batching`, contradicting the README's "default" claim. Restore
   the `recipe_options` block before any rebuild.

4. **`container/lemonade_test.sh` cannot pass against the published
   artifacts:** it never sets `-e MODEL`, and the image's baked default is the
   old filename `Qwen3-8B-F8E4M3.gguf`, so the model never registers (verified:
   health OK, empty model list, chat 422). Add
   `-e MODEL=/models/Qwen3-8B-Quark-F8E4M3.gguf` (and drop the hardcoded
   `IMG=roc8-lemonade:tr713`, which forces a manual `podman tag` first).

5. **docker.io and quay.io tags are not anonymously pullable** (both 401).
   README §4 already hedges ("images may not be pushed to every registry
   yet"), but §5 lists all three registries as places to get it; ghcr is the
   only one that works.

6. Minor: README §4's serve command binds host port 13305 verbatim and fails
   with `address already in use` on any box already running Lemonade
   (`lemonade_test.sh` itself anticipates this by using 13405). Worth one line
   in the common-failures table.

## Appendix — Q1_0 / Q2_0 modifications in the `roc8` fork

Inventory of the binary/ternary (PrismML Bonsai) support in
`The-Monk/llama.cpp` @ `3114c471f` (branch `roc8`), for extraction upstream:

**Core type plumbing**
- `ggml/include/ggml.h:431-432` — `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`
- `ggml/src/ggml-common.h:~195-209` — `block_q1_0` (128 elem / 18 B, fp16
  scale + 1-bit {-1,+1}) and `block_q2_0` (128 elem / 34 B, fp16 scale +
  2-bit ternary), with size `static_assert`s
- `ggml/src/ggml.c` — type traits (`:674`), `GGML_FTYPE_MOSTLY_Q2_0` mapping
  (`:1474`), quantize dispatch (`:7804`)
- `ggml/src/ggml-quants.c` — reference quantize/dequantize/vec-dot, plus the
  `ggml_validate_row_data` cases (`:6056` Q1_0, `:6060` Q2_0 — the 2026-07-26
  Q2_0 validation fix; required for `llama-quantize --type Q2_0` and
  `--check-tensors`)
- `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ops.cpp` — CPU paths

**CUDA/HIP (the RDNA4 work)**
- `mul_mat_q2_0_gemv.cu/.cuh` — dedicated Q2_0 decode GEMV (VALU-bound)
- `mul_mat_q2_0_wmma.cu/.cuh`, `mul_mat_q2_0_fp8route_mmq.cu/.cuh` — Q2_0
  prefill variants
- `mul_mat_q2_0_hipblaslt.cu/.cuh`, `mul_mat_q1_0_hipblaslt.cu/.cuh` —
  env-gated hipBLASLt prefill routes (`GGML_HIP_Q{1,2}_0_HIPBLASLT_*`)
- `template-instances/mmq-instance-q1_0.cu`, `mmq-instance-q2_0.cu` — MMQ
- shared-infrastructure edits: `mmvq.cu`, `vecdotq.cuh`, `convert.cu`,
  `dequantize.cuh`, `getrows.cu`, `mmq.cu/.cuh`, `common.cuh`, `ggml-cuda.cu`

**Other backends**
- OpenCL: `gemm_noshuffle_q1_0_f32.cl`, `gemv_noshuffle_q1_0_f32.cl`,
  `mul_mm_q1_0_f32_l4_lm.cl`, `mul_mv_q1_0_f32.cl`, `mul_mv_q1_0_f32_flat.cl`
- Vulkan: `vulkan-shaders/dequant_q1_0.comp`
- Metal: references in `ggml-metal-device.cpp`

Stale doc at the fork root: `Q1_0-INT4-ROUTES-DRAFT.stale-jul30.md` — marked
stale in its own filename; prune or refresh before pointing PrismML at the
branch.
