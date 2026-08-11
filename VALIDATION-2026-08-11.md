# Re-validation + format benchmarks — 2026-08-11

Follow-up to [VALIDATION-2026-08-10.md](VALIDATION-2026-08-10.md). All three
ship-blockers from that pass are fixed, the image was rebuilt from the current
tree and pushed, and the fixed image was benchmarked across the weight formats
on larger models (27B/35B class; the 8B is covered in the previous report).

**Environment:** 2× AMD Radeon AI PRO R9700 (gfx1201), rootless Podman + crun,
image `ghcr.io/the-monk/the-rock8:rdna4-tr713`, build **`56860a8fa`** (config
digest `d8205c131772a…`), binaries built against TheRock ROCm 7.13.0.
Measured single-card DRAM roofline used throughout: **631 GB/s**.

## Fixes verified (repo commit `84ea9cb`)

| 2026-08-10 finding | Status |
|---|---|
| README §0 no-arg `bench` broken in published image | **Fixed & verified** — the quickstart now runs verbatim: `Device 0` + pp128 2986 ± 725, tg32 59.84 ± 0.86 |
| `llama-completion` missing from image | **Fixed** — present; `validate_image.sh` passes 6/6 ldd closure, zero `/opt/rocm` |
| `entrypoint.sh` defined `LLAMA_ARGS` but never passed it | **Fixed & verified** — registration log in the new image records `llamacpp_args: -ngl 999 --cont-batching` |
| `lemonade_test.sh` could not pass against published artifacts | **Fixed** — sets `-e MODEL`/`-e MODEL_NAME`, `IMG` defaults to the ghcr image |
| Containerfile unbuildable (chased a moved `backend_versions.json`) | **Fixed** — version marker + slot path now derived from the lemonade SDK's own python API; the silent-swap verification passed in the build log |
| Image older than repo (f36df0f3c vs roc8 HEAD) | **Closed** — image build == submodule pin == public `roc8` HEAD == `56860a8fa` |

Lemonade round-trip on the new image: model auto-registered, chat answered
"Raisin", fingerprint `b9965-56860a8fa`. Perplexity is unchanged across the
rebuild (8B: 10.9345 ± 0.48054, identical to the 2026-08-10 measurement).

## Format × model benchmarks (larger models)

`llama-bench -p 512 -n 128 -ngl 999 -r 3` inside the published image, stock
settings — **no opt-in env levers enabled**. Two-GPU rows are layer-split
(`-ngl 999` across both cards); their roofline column is against the 1262 GB/s
aggregate. "Decode GB/s" = tg128 × file bytes (a proxy that is only meaningful
for dense models; MoE rows read only their routed experts per token).

| Model | Format | File | GPUs | pp512 t/s | tg128 t/s | Decode GB/s | % roofline |
|---|---|---|---|---|---|---|---|
| Qwen3.6-27B dense | **F8E4M3** | 29.26 GiB | 2 | 1188.4 ± 88.2 | 18.43 ± 0.07 | 579 | 46% agg |
| Qwen3.6-27B dense | Q8_K_XL | 32.89 GiB | 2 | 1098.4 ± 75.5 | 16.10 ± 0.13 | 568 | 45% agg |
| Qwen3.6-27B dense | Q6_K_XL | 23.87 GiB | 1 | 702.9 ± 59.2 | 20.71 ± 0.03 | 531 | 84% |
| Qwen3.6-27B dense | Q5_K_XL | 18.65 GiB | 1 | 887.1 ± 94.4 | 24.20 ± 0.06 | 485 | 77% |
| Bonsai-27B | **Q2_0** ternary | 6.66 GiB | 1 | 1148.9 ± 154.7 | 49.62 ± 0.28 | 355 | 56% of bytes; **~65% of the 76.9 t/s model roofline** (VALU-bound) |
| Bonsai-27B | **Q1_0** binary | 3.53 GiB | 1 | 1130.5 ± 151.1 | 43.96 ± 0.25 | 167 | 26% — VALU-bound, documented slower than Q2_0 |
| Qwen3.6-35B-A3B MoE | **F8E4M3** | 36.02 GiB | 2 | 3014.3 ± 10.8 | 74.54 ± 1.02 | — | overhead-bound (see below) |
| Qwen3.6-35B-A3B MoE | MXFP4 | 20.21 GiB | 1 | 3127.4 ± 35.3 | 74.36 ± 0.64 | — | " |
| Qwen3.6-35B-A3B MoE | Q4_K_M | 20.60 GiB | 1 | 2948.1 ± 13.0 | 75.13 ± 0.55 | — | " |
| Ornith-1.0-35B (A3B MoE) | **F8E4M3** | 34.24 GiB | 2 | 2624.6 ± 165.4 | 78.39 ± 1.35 | — | " |
| Ornith-1.0-35B (A3B MoE) | MXFP6 | 26.44 GiB | 2 | 747.7 ± 8.6 | 41.21 ± 1.43 | — | stock path; the MXFP6 hipBLASLt prefill route is opt-in and NOT enabled here |
| DeepSeek-R1-14B | **F8E4M3** | 15.98 GiB | 1 | 2525.7 ± 291.8 | 35.39 ± 0.15 | **607** | **96%** |

Failed row: a local `Ornith-1.0-35B-Q4_K_M.gguf` would not load — it is a
truncated quant artifact (`missing tensor 'blk.40.attn_norm.weight'`), not a
published model; nothing about the image is implicated.

**Format coverage:** F8E4M3 (dense + MoE experts), Q1_0, Q2_0, Q4_K, Q5_K,
Q6_K, Q8 K-family, MXFP4, MXFP6 — all load and run in the published image.
Not exercised here: F8E5M2, MXFP8, NVFP4 (no large local checkpoints), and the
model-blocked IU4/2:4-sparse paths (see README §2).

## Decode-capture analysis (against a 98%-of-roofline bar)

1. **fp8 dense on one card is the reference point: 96%** (14B). The format and
   the decode path are capable of near-roofline; every gap below is
   configuration or workload structure, not the fp8 kernel.
2. **Biggest available capture: the dedup lever is in this image and off.**
   `GGML_HIP_DEDUP_MMVQ_QUANT` (+`_BATCH`) is verified lossless across
   Q2_0/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/F8E4M3/IU4 and takes Q2_0 from ~65% to ~87%
   of its model roofline (51 → 67 t/s under MTP on this hardware). It shipped
   opt-in out of caution; recommending it become the appliance default is the
   single cheapest capture on this table.
3. **Two-GPU layer-split decode reads as ~45% aggregate — that is ~92%
   per-card equivalent.** The serialization is structural (each card idles
   while the other's layers run). The captures that exist in this image today
   are MTP self-speculative decode (27B fp8 18 → 45 t/s documented) and
   `lookup` mode; the async two-GPU pipeline (+75%) is still
   `speculative-simple`-only.
4. **MoE decode is pinned at ~74–78 t/s regardless of format** (fp8 = MXFP4 =
   Q4_K within noise): expert routing/launch overhead, not bandwidth. Capture
   requires expert-path fusion/graph work — real kernel engineering, not a
   flag.
5. **Q1_0 at 26% is the documented VALU-density result** — binary loses to
   ternary; the README already recommends Q2_0. The only real capture is an
   IU4-style repack onto the 95–97%-roofline int4 dot8 path.
6. **Q5_K/Q6_K (77–84%)**: candidates for the same rows-per-block/nwarps sweep
   Q4_K received (+2.9%), plus dedup applies.
