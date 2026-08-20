# Post-judging changelog

The AMD AI DevMaster Hackathon concluded on 19 August 2026. The submission
state stays pinned under the immutable `hackathon-2026-07-submission` tags on
this repo and the kernel fork; this file tracks what moved after the freeze.

## Unfrozen 20 August 2026

- **`MUL_MAT_ID` out-of-bounds write fix** (`63a9969bd1`): the mmvq ids-path
  row guard wrote out of bounds for quantized types with `rows_per_block > 1`
  when `nrows % rows_per_block != 0` — severely wrong MoE decode results with
  1-bit/ternary/fp8 experts. Fixed, gated, bench-neutral (60.56 t/s
  unchanged). This was found the day of the freeze and deliberately held so
  the judged artifact stayed immutable.
- **Hardened container entrypoint** (`e70c26e4d8`): model-mount guard inside
  the dispatch (`bash` and model-less `serve` keep working), `LLAMA_ARGS`
  wired explicitly through `recipe_options.llamacpp_args`, and the
  presence-based dedup env vars translated so `-e GGML_HIP_DEDUP_MMVQ_QUANT=0`
  actually disables.
- Image rebuilt from that tree, validated (library-closure check, in-image
  bench, perplexity exact to the 8/10 reference, lemonade round-trip), and
  published over the `rdna4-tr713` / `latest` tags.

## Measured deltas since submission

Same hardware, same models; the wider work lives in the living tree
([`rock8-release`](https://github.com/The-Monk/llama.cpp/tree/rock8-release))
and the PrismML collaboration (PrismML-Eng/llama.cpp#116).

| lane | submission era | frozen appliance (8/11 image) | current stack | with MTP heads |
|---|---|---|---|---|
| Bonsai-27B Q1_0 decode | 43.96 t/s | 62.2 | 66.6–68.5 | **97.4** (code, n=3) |
| Bonsai-27B Q2_0 decode | ~49.7 | ~51 | 51.8 | **77.5** (code, n=3) |
| Q1_0 prefill pp2048 | — | — | 1332–1359 (+SWAR) | — |
| CPU 8B Q2_0 decode (dual E5-2699v4) | 1.81 t/s | — | **35.55** (repack + NUMA + t=56) | — |
| fp8-8B in-image tg32 | 59.6–60.0 | 63.79 (8/18 re-run) | — | — |

Instruments that did not exist at submission: bench-card v2 (anchor + config
binding), empirical read-roofline calibration (640 GB/s measured), route-model
pre-registration, an ISA datapath grid (17 architectures × 32 instructions),
a PC-sampling protocol for gfx1201, per-tensor 2:4 sparsity sensitivity maps,
and the 2OF4_T1 sparse-ternary type (1.625 bpw, kernel-gated 43/43).
