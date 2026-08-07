# The Rock8 - Got any weights? 💪🦆
### RDNA4 (gfx1201) native-fp8 llama.cpp + Lemonade appliance

The Rock8 is a fork of llama.cpp/ggml that adds **native low-precision matrix
kernels for AMD RDNA4** (gfx1201 - Radeon AI PRO R9700, RX 9070 / 9070 XT,
W-series), packaged as a one-command rootless-Podman appliance on the
AMD TheRock ROCm 7.13 toolchain.

Everything below is **validated on real gfx1201 hardware** (dual R9700), against
the RDNA4 ISA, not inferred from benchmarks alone. Where a capability is *not*
yet usable, we say so plainly and explain what would unblock it.

**Kernel source:** [The-Monk/llama.cpp, branch `roc8`](https://github.com/The-Monk/llama.cpp/tree/roc8)

---

## The Rock8 - RDNA4 fp8 models (Hugging Face)

Native fp8 E4M3 GGUFs, each Quark-quantized from full-precision BF16 and
validated on gfx1201 (load + bench + PPL + coherence). Grab any one and the
appliance below runs it.

| Model | Source (license) | Params | GGUF | PPL | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|---:|---:|
| [**Quacken-8B-FP8**](https://huggingface.co/Gorilla4X/Quacken-8B-FP8) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0) | 8B | 9.2 GB | 10.93 | 4688 | 58.0 |
| [**Quacken-R1-14B-FP8**](https://huggingface.co/Gorilla4X/Quacken-R1-14B-FP8) | [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B) (MIT) | 14B | 16 GB | 8.97 | 2499 | 33.4 |
| [**Quacken-27B-FP8**](https://huggingface.co/Gorilla4X/Quacken-27B-FP8) | Qwen3.6-27B (Apache-2.0) | 27B | 29 GB | 7.14 | 1251 | 18.52 |
| [**Quacken-35B-A3B-FP8**](https://huggingface.co/Gorilla4X/Quacken-35B-A3B-FP8) | Qwen3.6-35B-A3B MoE (Apache-2.0) | 35B-A3B | 38.7 GB | 6.63 | -- | -- |
| [**Quacken-Ornith-35B-FP8**](https://huggingface.co/Gorilla4X/Quacken-Ornith-35B-FP8) | Ornith-1.0-35B (Apache-2.0) | 35B | 37.8 GB | 6.70 | -- | -- |

- **Collection:** [The Rock8 - RDNA4 fp8](https://huggingface.co/collections/Gorilla4X/the-rock8-rdna4-fp8-6a547070f667cb41db0bc2ed) (all models, one place).
- **All 5 live.** Each Quark-quantized from full-precision BF16, validated on gfx1201.

> PPL is wikitext, 20 chunks, `n_ctx=512`. Prefill/decode are `llama-bench` on
> gfx1201 (R9700); the 27B decode figure is 2-GPU (tensor-split).

---

## 1. Shipped features - what they do and when to use them

### Precision / matrix kernels
| Feature | What it is | Use case |
|---|---|---|
| **fp8 E4M3 weights** | Native RDNA4 WMMA fp8 for prefill + `v_dot4_f32_fp8_fp8` for decode | The default 8-bit format. No-compromise quality-per-byte; prefill ~+42% vs Vulkan. Use for any dense model where you want fp8. |
| **fp8 E5M2 (bf8) weights** | Native bf8 WMMA + decode | Wider dynamic range than E4M3 (5 exp bits) at the cost of ~+2.5% PPL (2 mantissa bits). Use when activations/outliers need the range. |
| **fp8 KV-cache** (`-ctk/-ctv f8e4m3`) | fp8 K and V cache in flash-attn | Halves KV memory -> longer context or more concurrent sequences on a 32 GB card. |
| **fp8 MoE** (`MUL_MAT_ID`) | fp8 for mixture-of-experts expert matmuls | fp8 for MoE models (e.g. Qwen3.6-35B-A3B). Quantizes the *experts*, not just attention. |
| **MXFP8** (`block_mxfp8`, e8m0 group-32 scale) | OCP Microscaling fp8; T77 hardware-dot2 decode | Ingests MLX `mx.quantize(mode="mxfp8")` models (e.g. OsaurusAI). T77 makes it the fastest fp8 *decode* path. |
| **2:4 structured-sparse fp8** (`2OF4_FP8`, SWMMAC) | RDNA4 2:4 sparse-tensor-core fp8 | Trained-2:4-sparse models (e.g. Sparse-Llama) at fp8 - 5.5 bpw, sparse-tensor-core throughput. |
| **2:4 structured-sparse fp16** (`2OF4_F16`, SWMMAC) | RDNA4 2:4 sparse f16 A/B | Trained-2:4-sparse models with **no quantization loss** - full-precision values on the sparse tensor cores. |

### Every weight format the driver reads

Complete list of what `roc8` decodes, with the block layout each one actually
uses. Everything here is dispatchable — these are `GGML_TYPE_*` entries, not
aspirations.

| Format | Block | Scale | Leaf encoding |
|---|---|---|---|
| **Q1_0** | 128 elem / 16 B | fp16 per block | 1 bit, {-1,+1} |
| **Q2_0** | 128 elem / 32 B | fp16 per block | 2 bit ternary, {-1,0,+1} |
| **Q4_0** | 32 elem / 16 B | fp16 per block | 4 bit unsigned, value = n-8 |
| **Q5_0** | 32 elem / 20 B | fp16 per block | 4 bit + a 5th bit-plane in `qh`, value = n-16 |
| **Q8_0** | 32 elem / 32 B | fp16 per block | signed int8 |
| **Q2_K … Q6_K** | 256-elem superblocks | multi-level | upstream ggml K-quants, unmodified |
| **IU4** | packed uniform int4 | — | native `v_wmma_i32_16x16x32_iu4` W4A4; dormant, see §2 |
| **F8E4M3** | 32 elem / 34 B | fp16 per block | OCP e4m3fn — 4 exp / 3 mantissa, bias 7 |
| **F8E5M2** | 32 elem / 34 B | fp16 per block | OCP bf8 — 5 exp / 2 mantissa, real ±Inf and NaN |
| **MXFP4** | 32 elem / 17 B | **UE8M0**, 2^(e-127) | OCP E2M1 — 2 exp / 1 mantissa, nibble-packed |
| **MXFP6** | 32 elem / 25 B | **UE8M0**, 2^(e-127) | OCP E3M2 — 3 exp / 2 mantissa, 4 values per 3 bytes |
| **MXFP8** | 32 elem / 33 B | **UE8M0**, 2^(e-127) | OCP e4m3fn, same leaf as F8E4M3 |
| **NVFP4** | 64 elem / 36 B | **UE4M3 ×4**, one per 16 | E2M1 leaf with finer-grained sub-block scales |
| **2:4-sparse fp8 / fp16** | SWMMAC fragments | — | structured sparsity on the sparse tensor cores |

Three scale conventions appear above and they are not interchangeable: a
continuous **fp16** delta (the Q-series and F8E*), a shared power-of-two
**UE8M0** exponent byte (the MX series, OCP microscaling), and **UE4M3**
per-sub-block scales (NVFP4). A decoder that assumes fp16 will silently
mis-scale every MX tensor it touches.

> **On "AMD FP4":** there isn't one, and we do not ship a format by that name.
> AMD's 4-bit path is **MXFP4**, the OCP Microscaling standard. **NVFP4** is
> NVIDIA's variant of the same E2M1 leaf with finer-grained UE4M3 scales; we read
> it so NVIDIA-quantised checkpoints load, not because it is an AMD format.

### Low-bit weight formats

| Feature | What it is | When to use it |
|---|---|---|
| **int4 decode** (`k_mmvq_dot8_iu4`) | Block-per-row GEMV on RDNA4's native `v_dot8_i32_iu4` 8-wide int4 dot | The fastest decode path we have: **96-97% of the measured 631 GB/s DRAM roofline**, correctness-gated against a CPU reference. Memory-bound and saturating -- there is no headroom left to take. |
| **Q2_0 ternary** (`block_q2_0`, 128/block) | 2-bit ternary {-1,0,+1} + fp16 block scale, with a dedicated GEMV | Natively-ternary models (Bonsai). ~2 bpw. Decode is VALU-bound rather than bandwidth-bound at this width. |
| **Q1_0 binary** (`block_q1_0`, 128/block) | 1-bit {-1,+1} + fp16 block scale | Measured **slower** than Q2_0 on Bonsai-27B (150 vs 164 t/s) -- below ternary the byte saving loses to VALU density. Shipped for completeness; Q2_0 is the better default. |
| **MXFP6** (E3M2 + UE8M0) | 6-bit OCP microscaling, 4 values per 3 bytes | The one MX width where byte-reduction still buys decode throughput (+17%). |

### Prefill routing

| Feature | What it is | When to use it |
|---|---|---|
| **Per-format hipBLASLt routes** | Dedicated prefill paths for Q1_0, Q2_0, Q4_K, Q8_0, F8E4M3, MXFP8, MXFP6, IU4, F16 -- each dequantises to int8 or fp8 and dispatches to Tensile | Large-batch prefill. **All are opt-in and env-gated**, and default to the stock kernel, for the reason below. |
| **Self-tuning algo cache** | Per-shape algorithm selection, disk-persisted, working around gfx1201's missing cost model | +15.6% pp1024 on Q2_0 once the cache is warm. Novel shapes pay a one-off tuning cost. |

> **Why the routes are opt-in.** The gate keys on the batch dimension M against a scalar threshold, but the win is **shape-dependent**: the same route and threshold gives **+32.7% on an 8B model and -38.2% on a 24B** at overlapping batch sizes. The crossover moves at least 4x with model shape, so a threshold correct for one model is a large regression for another. We would rather ship a conservative default than a fast one that is wrong off-box. Measured in `benchmarks/crossover.sh` of the hackathon repo below.

### The prefill routes, and how to drive them

Nine per-format hipBLASLt routes. Each dequantises the weight to int8 or fp8 and
dispatches to Tensile, and each is **off by default** -- see the shape-dependence
note below for why. Every route exposes the same six knobs:

| env var | effect |
|---|---|
| `GGML_HIP_<FMT>_HIPBLASLT_PREFILL=1` | enable the route at all |
| `GGML_HIP_<FMT>_HIPBLASLT_MTHRESH=<n>` | minimum batch M before the route engages; below it, the stock kernel runs |
| `GGML_HIP_<FMT>_HIPBLASLT_FP8=1` | dequantise to fp8 instead of int8 -- more accurate for higher-bpw formats, since fp8's exponent preserves per-block range that int8 collapses |
| `GGML_HIP_<FMT>_HIPBLASLT_TUNE_CACHE=<path>` | where to persist the per-shape algorithm choice |
| `GGML_HIP_<FMT>_HIPBLASLT_NOTUNE=1` | skip tuning; use the heuristic pick |
| `GGML_HIP_<FMT>_HIPBLASLT_WCACHE_MB=<n>` | cap the dequantised-weight cache |

`<FMT>` is one of **`Q1_0` `Q2_0` `Q4_K` `Q8_0` `F8E4M3` `MXFP6` `MXFP8` `IU4` `F16`**.
Global escape hatch: `GGML_HIP_NO_HIPBLASLT=1` disables all of them.

The self-tuning cache exists because gfx1201 has no hipBLASLt cost model -- the
heuristic mis-picks badly (163 vs 318 TOP/s on an ffn-down shape). The cache
stores the *index*, not the algorithm blob; restoring a blob segfaults.

> **F16 is disabled deliberately, and it is not a bug to be fixed.** Every other
> route earns its win by *owning the dequantisation* -- it decides how the weights
> reach the tensor cores (int8 or fp8, which is what the `_FP8` knob selects), and
> that choice is the mechanism. **F16 has no dequantisation to own.** The weights
> are already in a format the tensor cores consume directly, so our route and the
> stock `hipblasGemmEx` path converge on the same Tensile kernels; ours simply adds
> plan-cache and tuning overhead on top.
>
> A better weight converter cannot recover this. We checked the obvious
> candidate -- our route casts F32 activations to F16 with its own kernel -- but
> the baseline does exactly the same thing (`src1_as_f16` / `to_fp16_cuda` in
> `ggml-cuda.cu`), so the cast is a wash, not an overhead we are paying alone.
> The absent `_FP8` knob on this route is the tell: there is no intermediate
> format decision to make. The flag exists for measurement, not for use.

### Decode-side levers

| env var | what it does |
|---|---|
| `GGML_HIP_DEDUP_MMVQ_QUANT=1` | share identical weight reads across a decode step (also `_BATCH`, `_FP8`) |
| `GGML_HIP_IU4_MMQ=1` | int4 MMQ path |
| `GGML_HIP_F16_DOT2_DECODE=1` | `v_dot2_f32_f16` decode -- measured a **loss** single-issue; kept for reproducibility |
| `GGML_HIP_F8E5M2_DOT4=1` | bf8 decode dot |
| `GGML_HIP_FUSE_MMVQ_QUANT=1` | fuse activation quantisation into the GEMV |
| `GGML_HIP_BINARY_GEMV_NWARPS` / `_ROWS_PER_BLOCK` | launch geometry for the Q1_0 binary GEMV |

Roughly **123 `GGML_HIP_*` flags** exist in total, most of them experiment
switches for the 2:4-sparse and dense-fp8 MMQ variants (`_V2`, `_V3`,
`_ADAPTIVE`, `_ILP`, `_GEOM`, `_KPS`, `_SHAPE_BENCH`). They are research
scaffolding, not a supported interface -- the table above is what is meant to be
driven.

### What is actually in the fork

32 kernel sources that upstream llama.cpp does not have:

- **decode** -- `mmvq_iu4.cu`, `mmvq_dot2f16.cu`, `mul_mat_q2_0_gemv.cu`, `mul_mat_iu4_gemv.cu`
- **prefill routes** -- `mul_mat_{q1_0,q2_0,q4_K,q8_0,f8e4m3,mxfp6,mxfp8,iu4,f16}_hipblaslt.cu`
- **dense fp8 MMQ** -- `mul_mat_dense_fp8_mmq.cu`, `mul_mat_dense_fp8_v3.cu`
- **2:4 sparse** -- `mul_mat_2of4_{f16,fp8,fp8_mmq,iu4_k64}.cu`, `swmmac24{,_iu4_fixed,_iu4_k64}.cu`
- **int4 W4A4** -- `iu4_w4a4.cu`, `mul_mat_iu4{,_mmq,_ck_wmma}.cu`
- **infrastructure** -- `hipblaslt_wcache.cu` (weight-cache invalidation), `quantize_fp8_sr.cu` (stochastic rounding), `mxfp8_selftest.cu`, `dot2_bf16_probe.cu`

### Correctness fixes (not performance -- just bugs)

| Fix | What was wrong |
|---|---|
| **`GGML_TYPE_Q2_0` validation** | Q2_0 was missing from `ggml_validate_row_data`, which silently broke `llama-quantize --type Q2_0` **and** `--check-tensors` for every Q2_0 GGUF. We audited the rest of the table; Q2_0 was the only gap. |
| **Weight-cache invalidation** | The hipBLASLt weight cache was not invalidated when a buffer was freed, so a reused allocation could serve stale weights. Each route now registers an invalidator; zero hot-path cost. |
| **CK 2:4-sparse SWMMAC** | Three bugs in Composable Kernel's gfx1201 sparse path (phantom fill on <2-nonzero groups, `pk_int4_t` byte-vs-nibble OOB, index swap+XOR-1). Error 352 -> 0. Submitted upstream as [ROCm/composable_kernel#3759](https://github.com/ROCm/composable_kernel/pull/3759). |

### Decode / serving levers
| Feature | What it is | Use case |
|---|---|---|
| **MTP self-speculative decode** (`--spec-type draft-mtp`) | Uses the model's own next-token (MTP) head as the draft | Single-GPU interactive latency - decode **95 t/s > Vulkan 91** on Qwen3.6-27B. The single-GPU champion. |
| **DFlash spec-decode** (`--draft-dflash`) | Q8/fp8 DFlash drafter + fp8 target | 1-GPU 52 t/s while leaving the 2nd GPU free for an agent fleet. |
| **Async spec-decode pipeline** (`LLAMA_SPEC_ASYNC=2`) | Draft-gen ‖ verify on **separate GPUs** (disjoint compute); needs a draft model + 2 cards | **A 2-GPU RDNA4 box serving one latency-critical stream** — a local coding assistant, an interactive chat, or an agent loop where you want the highest tokens/sec and have both cards. The draft model sits on GPU1 generating candidates *while* GPU0 verifies the previous batch, so neither card idles → **+75% decode vs running them serially**. Single-GPU is a wash (the two saturating kernels time-share one card), so this is specifically a *dual-card* lever; auto-enable when 2 GPUs + a draft are detected is on the roadmap. |
| **Weight dedup + verify-dedup** | Identical weight reads across a decode step are issued once and shared, including across the speculative-decode verify batch | Takes Q2_0 MTP decode from a raw ~51 t/s to **67.18 t/s = 87% of the 76.9 t/s roofline for that model**. Lossless, verified across Q2_0/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/F8E4M3/IU4. Automatic. |
| **Register-spill fixes** | fattn-vec fp8-KV (hd128/256, ncols=2) -> 0 spill; mmq fp8/bf8/mxfp8 48-tile -> 0 spill | Removes VGPR spills in fp8-KV + spec-decode/batched-verify, and in specific prefill batch widths (+17.8% on the 48-tile). Automatic - no flag. |
| **Continuous batching -- multi-user serving** (`--cont-batching`, default) | Merges concurrent decode steps: one weight read advances **N** sequences (the amortization that makes serving fast) | **Serve a handful of concurrent users/agents off one card -- the vLLM replacement on RDNA4.** Aggregate throughput scales **4.06x at npl=8** on the 35B-A3B fp8 MoE, realizing the same batching amortization that was vLLM's "server win" -- but *natively* and *with fp8* (vLLM silently dequants fp8 on gfx1201, so its win evaporates here). vLLM's KV-memory edge is **already halved on The Rock8 by fp8 KV-cache** (`-ctk/-ctv f8e4m3` -> ~2x the sequences/GB, on top of the fp8 weights) -- the only remaining piece is PagedAttention's fragmentation-free *paging*, which is landing in llama.cpp (#21961). |

### Operations
| Feature | What it is | Use case |
|---|---|---|
| **Auto-ctx / OOM-guard** (`roc8-autoctx`) | Sizes context to free VRAM (model + KV, fp8-KV-aware); CPU-spill + desktop/GDM headroom aware; caps `-ngl` <= cores-1; clamps API-forced ctx; `--bypass-oom` escape | Prevents the classic "load OOMs the card" failure. The Lemonade default so users never hand-tune ctx. Rejects an oversized API `ctx_size` and forces it back to the computed max (or errors, your choice). |
| **The Lemonade appliance** | Rootless-Podman image (`ubuntu:24.04` + TheRock 7.13, one extra dep: `libatomic1`) | One-command portable RDNA4 AI stack. Proven to run pure-7.13 with **zero `/opt/rocm`** on the host. Requires `crun` for GPU passthrough (not `runc`). |

---

## 2. Dormant / model-blocked capabilities - honest status

These are **hardware-validated and correct**, but not usable end-to-end today
because no producer *model* exists (or an accuracy gate wasn't met). We ship
them dormant-complete - correct-when-a-model-exists - rather than pretend
they're ready. This is deliberate: the driver is complete for the *hardware*,
not just for today's models.

| Capability | State | Why it's blocked | What would unblock it |
|---|---|---|---|
| **iu4 W4A4** (`block_iu4`, native int4xint4 WMMA) | Kernel correct -- the transposed-accumulator readback bug is now fixed in BOTH the selftest AND the real-model matmul (`mul_mat_iu4`, `DATA_LAYOUT_J_MAJOR`); selftest passes exact | **Model-blocked** -- no packed uniform-int4 **W4A4** model exists that avoids an *online* activation rotation | A W4A4 model whose rotation/smoothing folds **offline** into weights (co-trained BitNet-a4.8, or QuaRot/SpinQuant export with rotation pre-applied) |
| **INT6 compressed all-reduce** (dual-GPU tensor-parallel) | **Codec correct, CPU/numpy-validated**; the HIP two-shot kernel compiles but has not been run end-to-end | **Not measured on the card.** The 2.46x figure is a *byte ratio* (0.8125 B/elem vs fp16's 2), computed in numpy -- not a throughput measurement. The exchange also has to route around the RCCL gfx1201 tuning gap that deadlocks AllReduce without `NCCL_PROTO=Simple` | Run the two-shot kernel end-to-end on both cards and measure against an uncompressed baseline. Until then this is a design with a validated codec, not a shipped lever |
| **Weight-only int4 (W4A16)** -- Quark `int4_wo` -> ggml Q4 | **Path verified:** Quark 0.12 produces runnable weight-only int4 (`int4_wo_128/64/32`, **no online rotation**), and llama.cpp Q4_K already runs it | **Converter-blocked, *not* model-blocked** -- our converter ingests Quark **fp8** but not int4 yet; Quark GGUF export is Q4_1/toy-arch only | Extend the converter to ingest Quark `int4_wo` -> a ggml Q4 block. Unlike W4A4 this needs **no special model** -- pure plumbing. First use: the all-Quark spec-decode **draft** for the Omni |
| **Native bf8 `V_DOT4_F32_BF8_BF8`** | Built, ISA-validated (emits the real opcode), VGPR-efficient | **Accuracy-gated OFF** - bf8 activations cost +2.5% PPL vs the int8-activation dot2 path | A lower-error bf8 activation scheme (or a model tolerant of the loss); the kernel is ready to flip on |
| **2:4-sparse int8** (`swmmac_iu8`) | Kernel validated | **Model-blocked** - no W8A8 2:4-sparse model on hand | A trained/exported W8A8 2:4-sparse model + its converter |
| **2:4-sparse iu4** (`swmmac_iu4`) | Kernel validated | **Model-blocked** - needs the iu4 activation path *and* a 2:4-int4 model | A co-trained Sparse-BitNet-class 2:4-int4 model |
| **Mixed fp8xbf8 WMMA** (E4M3 weights x E5M2 activations, `GGML_HIP_FP8_ACT=bf8`) | Runtime toggle, correct | **Accuracy-gated OFF** by default (+0.29% PPL) | Kept as an opt-in runtime lever; flip `GGML_HIP_FP8_ACT=bf8` if the trade suits your model |

**Why keep dead-looking kernels?** RDNA4 silicon supports these paths
(verified against the ISA via `llvm-objdump --mcpu=gfx1201`, not header greps).
Shipping them correct-but-dormant means the day a suitable model appears, it
runs with zero kernel work. It also documents *exactly* what RDNA4 can and
cannot do for low-precision inference.

---

## 3. Roadmap / optional levers (not yet built)
- **Auto-enable the async pipeline** when 2 GPUs + a draft model are detected (today it's the `LLAMA_SPEC_ASYNC=2` opt-in).
- **mmq_x=48 spill** on F8E4M3's headroom - spot-check if future changes add register pressure.
- **MXFP4->FP8 upcast** converter (lossless upcast onto the fp8 tensor-core path).
- **Hardware-aware quantizer** - detect GPU capability -> pick the optimal quant -> validate on-box -> emit.
- **Multi-arch port** - capability-gated RDNA1/2/3/3.5 + CDNA (hardware-owner validated).

---

## 4. Run a model

### Lemonade appliance (container)
The appliance is a rootless-Podman image on TheRock ROCm 7.13 - no host `/opt/rocm`
needed. Intended pull paths (images may not be pushed to every registry yet):

```
# any of these resolve to the same image:
podman pull ghcr.io/the-monk/the-rock8:rdna4-tr713
podman pull docker.io/gorilla4x/the-rock8:rdna4-tr713
podman pull quay.io/the-monk/the-rock8:rdna4-tr713

# serve a downloaded Quacken GGUF (mount the model dir; crun is mandatory for GPU)
podman run -d --rm --runtime crun --name lemonade \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v /path/to/quacken-8b:/models:ro \
  -e MODEL=/models/Qwen3-8B-Quark-F8E4M3.gguf -e MODEL_NAME=Quacken-8B-FP8 \
  -e HIP_VISIBLE_DEVICES=0 -p 13305:13305 \
  ghcr.io/the-monk/the-rock8:rdna4-tr713 serve

curl -s http://localhost:13305/api/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"user.Quacken-8B-FP8","messages":[{"role":"user","content":"What do you call a dried grape? Answer in one word. /no_think"}],"max_tokens":16}'
```

See [`container/`](container/) for the Containerfile, entrypoint, and the full
portability writeup ([`container/README.md`](container/README.md)).

### llama.cpp directly
```
# prefill/decode bench
llama-bench -m Qwen3-8B-Quark-F8E4M3.gguf -ngl 99 -p 512 -n 128

# chat
llama-cli  -m Qwen3-8B-Quark-F8E4M3.gguf -ngl 99 -p "What do you call a dried grape? Answer in one word."

# 27B is 2-GPU (tensor-split across two 32 GB cards)
llama-bench -m Qwen3.6-27B-Quark-F8E4M3.gguf -ngl 999   # sees both R9700s
```

---

## 4b. A local model wrote some of these kernels

The fp8 E4M3 decode kernel shipped in our hackathon submission was **written by
Qwen-AgentWorld-35B-A3B running on the R9700 itself**, from a written
specification. It never saw the reference implementation, the inputs, or the
expected outputs. Graded blind against a golden reference it returned
`max_rel_err = 0.000000e+00` -- bit-identical across 4096 rows, first attempt --
and the compiled binary contains 8 emissions of `v_dot4_f32_fp8_fp8`, so the
hardware instruction really is used.

The same protocol was then run across seven quantisation formats:

| format | verdict | error |
|---|---|---|
| Q1_0 (1-bit binary) | REPRODUCED | **0** |
| Q2_0 (2-bit ternary) | REPRODUCED | 1.08e-06 |
| Q4_0 (4-bit linear) | REPRODUCED | 1.04e-06 |
| Q5_0 (5-bit + qh plane) | REPRODUCED | **0** |
| Q8_0 (8-bit signed) | REPRODUCED | **0** |
| MXFP4 (E2M1 + UE8M0) | REPRODUCED | 7.31e-07 |
| **MXFP8** (E4M3 + UE8M0) | **NOT REPRODUCED** | **2.087** |

Six of seven. MXFP8 is left in as a failure because it is the most informative
row: the model had already written a correct E4M3 kernel and a correct
UE8M0-scaled kernel, and MXFP8 is exactly those two combined -- capability on the
parts did not compose into capability on the whole.

Asked to write the *correctness gate* instead of the kernel, it failed outright:
its harness rejected all five test kernels including the correct one, across
three rounds. **It does the work well and judges it badly**, which is why every
result above was graded by a harness that owns the reference rather than by the
model.

Rig and captured runs: [Hyperloom](https://github.com/The-Monk/rocky-hackathon),
`kernels/decode/agent_repro/`.

## 5. Where to get it (every artifact links to the others)
- **Models:** the *The Rock8 - RDNA4 fp8* collection on Hugging Face (`Quacken-*-FP8`), under
  [Gorilla4X](https://huggingface.co/Gorilla4X) - each Quark-quantized from full-precision BF16,
  with PPL + throughput benches on gfx1201.
- **Container:** `ghcr.io` / Docker Hub / Quay.io - `the-rock8:rdna4-tr713`
  (podman *and* docker pull the same image; use `--runtime crun` for GPU).
- **Source:** this repo ([The-Rock8](https://github.com/The-Monk/The-Rock8)) - kernels, patch series,
  appliance recipe, and these docs.

*Land on any one, reach them all.*

---

## License

The Rock8 tooling and appliance recipes in this repo are MIT-licensed. The kernel
work itself lives in our llama.cpp fork, branch `roc8`:
**https://github.com/The-Monk/llama.cpp/tree/roc8** — which is a fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, (c) the ggml
authors). All credit for llama.cpp belongs upstream; none of this is upstreamed
and it carries no endorsement from the llama.cpp maintainers.
The published model weights are **derivatives** and carry their **source model's
license** - attributed on each model card:
Quacken-8B-FP8 (Apache-2.0, from Qwen/Qwen3-8B),
Quacken-R1-14B-FP8 (MIT, from DeepSeek-R1-Distill-Qwen-14B).
Coming: Quacken-27B-FP8 (Apache-2.0, from Qwen3.6-27B).
