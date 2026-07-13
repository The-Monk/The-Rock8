# ROC8 + Lemonade appliance — pure TheRock ROCm 7.13 (card 124 ship gate)

Containerized portability test proving the ROC8 (mxfp8 fork, `build-tr713`) + Lemonade
stack is genuinely self-contained on **TheRock ROCm 7.13 only**, with **zero `/opt/rocm-7.2.4`**.
Rootless Podman on Zorin-AI (Ubuntu 24.04 base, ZFS root, dual R9700 gfx1201).

## Result: YES — the full stack runs on pure TheRock 7.13.
- ldd closure: 100% resolved to the image's `/opt/therock` + `/opt/llama`, **zero `/opt/rocm`**.
- GPU: `llama-bench` runs on gfx1201 (R9700), matches host-forced-7.13 within noise.
- PPL: **byte-identical** to host-forced-7.13 (11.0282 ± 0.486) — the 7.2.4-rocBLAS-under-
  7.13-kernels host hybrid did NOT affect results (forced-7.13 == container-7.13).
- Lemonade OpenAI endpoint returns **"Paris"** through the rocm backend (our llama-server).

## Shippable base-image contents
- Base: `ubuntu:24.04`
- **Only extra OS package the binaries need: `libatomic1`.** TheRock bundles numa/drm/elf
  under `lib/rocm_sysdeps/`, and `libomp.so` under `lib/llvm/lib/` — so no libnuma/libdrm/
  libelf/libomp OS packages are required. (`python3`/`python3-venv`/`curl`/`ca-certificates`
  are for the Lemonade server layer only.)
- ROCm 7.13 sonames actually loaded (all from TheRock): `libamdhip64.so.7`, `librocblas.so.5`,
  `libhipblas.so.3`, `libhipblaslt.so.1`, `libamd_comgr.so.3`, `librocroller.so.1`,
  `librocm_sysdeps_{elf,drm,drm_amdgpu,numa}.so.*`.

## Critical host-vs-container runtime note (crun, not runc)
Rootless Podman here defaulted to **runc**, which silently ignores `--group-add keep-groups`
(a crun-only annotation) → `/dev/kfd` (group `render`/992) stays inaccessible → HIP reports
"no ROCm-capable device". **Fix: `crun` is installed; always run with `--runtime crun`.**

## Build
```
cd /home/jmonk/src/mainline-llama.cpp-mxfp8/container
# context already staged: therock/ (TheRock gfx1201-7.13.0 lib tree), llama/ (build-tr713 bin)
podman build --runtime crun -t roc8-lemonade:tr713 -f Containerfile .
```

## Run — Lemonade server (the appliance)
```
podman run -d --rm --runtime crun --name lemonade \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v /aipool/models/qwen3-8b-fp8:/models:ro \
  -e HIP_VISIBLE_DEVICES=0 -p 13305:13305 \
  roc8-lemonade:tr713 serve
# model auto-registered as user.Qwen3-8B-FP8, backend=rocm
curl -s http://localhost:13305/api/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"user.Qwen3-8B-FP8","messages":[{"role":"user","content":"Capital of France? one word /no_think"}],"max_tokens":16}'
```

## Run — bench / ppl directly (no bind-mounts except the model)
```
podman run --rm --runtime crun --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v /aipool/models/qwen3-8b-fp8:/models:ro roc8-lemonade:tr713 bench
# or:  ... roc8-lemonade:tr713 ppl -f /wiki/wiki.test.raw --chunks 20 -ngl 99   (mount /wiki)
```

## Measurements (Qwen3-8B-F8E4M3, GPU0 R9700 gfx1201)
| path                | pp128 (t/s)      | tg32 (t/s)   | PPL (20 chunks)    |
|---------------------|------------------|--------------|--------------------|
| host forced-7.13    | 2631.6 ± 552.7   | 63.51 ± 1.01 | 11.0282 ± 0.48605  |
| container (bind)    | 2588.2 ± 572.9   | 63.59 ± 0.90 | 11.0282 ± 0.48605  |
| built image         | 2634.9 ± 559.1   | 63.58 ± 1.09 | 11.0282 ± 0.48605  |
(pp128 ±variance is large because pp128 is a tiny prefill; tg/PPL are the stable signals.)

## Files
- `Containerfile` — the image recipe.
- `entrypoint.sh` — serve/bench/ppl/bash dispatch + Lemonade model-register & rocm wiring.
- `therock/` , `llama/` — staged build context (hardlinked from the real inputs).
- `validate_image.sh`, `lemonade_test.sh` — the from-image validation harnesses.

## Gaps / portability notes
1. **crun is mandatory** (see above) — document/install on any target host; runc will fail GPU.
2. `libatomic1` must be in the base image (only true extra OS dep).
3. Podman graphroot on ZFS needs `fuse-overlayfs` (host storage.conf set accordingly);
   the overlay kernel driver is unreliable on ZFS.
4. Model is NOT baked (mounted at `/models`), matching Lemonade's separate model-dir model.
5. Lemonade prefixes user-registered models with `user.` → request `user.Qwen3-8B-FP8`.
6. Lemonade's `lemonade-server-dev` (Python) prints a deprecation notice pointing to the C++
   server; functional but worth tracking for the shipped appliance.
