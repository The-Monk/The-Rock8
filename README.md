# The Rock8 - Got any weights? 💪🦆
### RDNA4 (gfx1201) native-fp8 llama.cpp + Lemonade appliance

The Rock8 is a fork of llama.cpp/ggml that adds **native low-precision matrix
kernels for AMD RDNA4** (gfx1201 - Radeon AI PRO R9700, RX 9070 / 9070 XT,
W-series), packaged as a one-command rootless-Podman appliance on the
AMD TheRock ROCm 7.13 toolchain.

Everything below is **validated on real gfx1201 hardware** (dual R9700), against
the RDNA4 ISA, not inferred from benchmarks alone. Where a capability is *not*
yet usable, we say so plainly and explain what would unblock it.

---

## The Rock8 - RDNA4 fp8 models (Hugging Face)

Native fp8 E4M3 GGUFs, each Quark-quantized from full-precision BF16 and
validated on gfx1201 (load + bench + PPL + coherence). Grab any one and the
appliance below runs it.

| Model | Source (license) | Params | GGUF | PPL | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|---:|---:|
| [**Quack-8B-FP8**](https://huggingface.co/Gorilla4X/Quack-8B-FP8) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0) | 8B | 9.2 GB | 10.93 | 4688 | 58.0 |
| [**Quack-R1-14B-FP8**](https://huggingface.co/Gorilla4X/Quack-R1-14B-FP8) | [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B) (MIT) | 14B | 16 GB | 8.97 | 2499 | 33.4 |
| [**Quack-27B-FP8**](https://huggingface.co/Gorilla4X/Quack-27B-FP8) ⏳ | Qwen3.6-27B (Apache-2.0) | 27B | ~29 GB | 6.81 | soon | soon |
| [**Quack-35B-A3B-FP8**](https://huggingface.co/Gorilla4X/Quack-35B-A3B-FP8) ⏳ | Qwen3.6-35B-A3B (MoE, Apache-2.0) | 35B-A3B | ~34 GB | soon | soon | soon |
| [**Quack-Ornith-35B-FP8**](https://huggingface.co/Gorilla4X/Quack-Ornith-35B-FP8) ⏳ | Ornith-1.0-35B (Apache-2.0) | 35B | ~34 GB | soon | soon | soon |

- **Collection:** [The Rock8 - RDNA4 fp8](https://huggingface.co/Gorilla4X) (all models, one place).
- ⏳ = uploading / final validation in progress — each row's HF link goes live as the model lands (27B re-validating the authentic load path; 35B-A3B MoE + Ornith finishing their converter fixes).

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

### Decode / serving levers
| Feature | What it is | Use case |
|---|---|---|
| **MTP self-speculative decode** (`--spec-type draft-mtp`) | Uses the model's own next-token (MTP) head as the draft | Single-GPU interactive latency - decode **95 t/s > Vulkan 91** on Qwen3.6-27B. The single-GPU champion. |
| **DFlash spec-decode** (`--draft-dflash`) | Q8/fp8 DFlash drafter + fp8 target | 1-GPU 52 t/s while leaving the 2nd GPU free for an agent fleet. |
| **Async spec-decode pipeline** (`LLAMA_SPEC_ASYNC=2`) | Draft-gen ‖ verify on **separate GPUs** (disjoint compute); needs a draft model + 2 cards | **A 2-GPU RDNA4 box serving one latency-critical stream** — a local coding assistant, an interactive chat, or an agent loop where you want the highest tokens/sec and have both cards. The draft model sits on GPU1 generating candidates *while* GPU0 verifies the previous batch, so neither card idles → **+75% decode vs running them serially**. Single-GPU is a wash (the two saturating kernels time-share one card), so this is specifically a *dual-card* lever; auto-enable when 2 GPUs + a draft are detected is on the roadmap. |
| **Register-spill fixes** | fattn-vec fp8-KV (hd128/256, ncols=2) -> 0 spill; mmq fp8/bf8/mxfp8 48-tile -> 0 spill | Removes VGPR spills in fp8-KV + spec-decode/batched-verify, and in specific prefill batch widths (+17.8% on the 48-tile). Automatic - no flag. |

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
| **iu4 W4A4** (`block_iu4`, native int4xint4 WMMA) | Kernel correct (transposed-readback bug fixed, selftest 20/20) | **Model-blocked** - no packed uniform-int4 W4A4 model exists that avoids an *online* activation rotation | A W4A4 model whose rotation/smoothing is folded **offline** into weights (so inference needs only a plain int4 activation cast), e.g. a co-trained BitNet-a4.8, or a Quark/QuaRot export with the rotation pre-applied |
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

# serve a downloaded Quack GGUF (mount the model dir; crun is mandatory for GPU)
podman run -d --rm --runtime crun --name lemonade \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v /path/to/quack-8b:/models:ro \
  -e MODEL=/models/Qwen3-8B-Quark-F8E4M3.gguf -e MODEL_NAME=Quack-8B-FP8 \
  -e HIP_VISIBLE_DEVICES=0 -p 13305:13305 \
  ghcr.io/the-monk/the-rock8:rdna4-tr713 serve

curl -s http://localhost:13305/api/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"user.Quack-8B-FP8","messages":[{"role":"user","content":"Capital of France? one word /no_think"}],"max_tokens":16}'
```

See [`container/`](container/) for the Containerfile, entrypoint, and the full
portability writeup ([`container/README.md`](container/README.md)).

### llama.cpp directly
```
# prefill/decode bench
llama-bench -m Qwen3-8B-Quark-F8E4M3.gguf -ngl 99 -p 512 -n 128

# chat
llama-cli  -m Qwen3-8B-Quark-F8E4M3.gguf -ngl 99 -p "The capital of France is"

# 27B is 2-GPU (tensor-split across two 32 GB cards)
llama-bench -m Qwen3.6-27B-Quark-F8E4M3.gguf -ngl 999   # sees both R9700s
```

---

## 5. Where to get it (every artifact links to the others)
- **Models:** the *The Rock8 - RDNA4 fp8* collection on Hugging Face (`Quack-*-FP8`), under
  [Gorilla4X](https://huggingface.co/Gorilla4X) - each Quark-quantized from full-precision BF16,
  with PPL + throughput benches on gfx1201.
- **Container:** `ghcr.io` / Docker Hub / Quay.io - `the-rock8:rdna4-tr713`
  (podman *and* docker pull the same image; use `--runtime crun` for GPU).
- **Source:** this repo ([The-Rock8](https://github.com/The-Monk/The-Rock8)) - kernels, patch series,
  appliance recipe, and these docs.

*Land on any one, reach them all.*

---

## License

The Rock8 tooling and appliance recipes in this repo are MIT-licensed (this repo
is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT).
The published model weights are **derivatives** and carry their **source model's
license** - attributed on each model card:
Quack-8B-FP8 (Apache-2.0, from Qwen/Qwen3-8B),
Quack-R1-14B-FP8 (MIT, from DeepSeek-R1-Distill-Qwen-14B).
Coming: Quack-27B-FP8 (Apache-2.0, from Qwen3.6-27B).
