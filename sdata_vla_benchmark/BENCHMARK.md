# SData frozen benchmark — protocol (read this before trusting the table)

## What “video” means in this codebase

**Training** (`training/train_sdata_attention.py`): for each clip, the dataset loads **one frame per camera** — the **middle frame** of each `.mp4` — plus **audio** (when present). It does **not** feed full frame sequences through the vision encoder.

**MMFuse eval** (`scripts/predict_with_model.py`): uses the **same** rule — **`_load_middle_frame`** for cam1 and cam2 — then fusion + classifier. So “trained on video files” here means **one spatial sample per video** (mid-frame), not a temporal video transformer.

So the meaningful comparisons are:

| Setting | Vision | Audio |
|--------|--------|-------|
| **MMFuse `vision_only`** | cam1 + cam2 mid-frames → fusion | zeros |
| **MMFuse `full`** | cam1 + cam2 mid-frames → fusion | 2.5 s wav |
| **Frozen CLIP / CLIP-RT / OpenCLIP** | see below | none |

## Frozen CLIP-style baselines: dual camera (default)

MMFuse always uses **two** camera streams at eval. A **single-cam** frozen baseline (cam1 only) was **easier to implement** but **not** vision-aligned with MMFuse.

**Default now:** `clip_rt` and `openclip_*` **average** per-class scores from **cam1 mid-frame** and **cam2 mid-frame** (same sampling as training/eval). This matches the **two-view** setup; it is still **not** temporal video modeling.

- Turn off: `export SDATA_FROZEN_DUAL_CAM=0` (single-cam ablation; uses cam1 unless you change CLI).
- Per-run CLI: `clip_rt_infer.py` / `openclip_zs_infer.py` support `--dual-cam` / `--no-dual-cam`.

## What is still *not* matched

1. **Audio:** Only **MMFuse `full`** uses audio. Frozen models have **no** mic signal. For a **vision-only** fair row, compare **`mmfuse_vision_only`** to **CLIP baselines**, not necessarily to **`mmfuse_full`**.

2. **Fusion:** MMFuse learns **cross-modal attention** between the two views (and audio in full mode). Frozen baselines use a **simple average of probabilities** across views — a weak fusion upper bound for generic VL models, not learned fusion.

3. **HF text VLAs (`openvla` slot):** Still **one image** per forward in this harness (unless extended). Dual-cam averaging for generative VLAs is not implemented here.

4. **Domain:** Frozen models are **not** trained on SData; MMFuse is. Low frozen accuracy can still reflect **distribution shift**, not only protocol bugs.

## RT-1 / RT-2 (`rt1`, `rt2`) and SayCan (`saycan`) — **optional** `--models` keys

These are **not** in the default benchmark list (they do not add 8-class metrics here). Pass **`rt1` `rt2` `saycan`** explicitly if you want JSON rows for documentation — with honest limits:

- **`rt1` / `rt2`:** Not Google’s unreleased RT checkpoints. They **download [RDT](https://huggingface.co/robotics-diffusion-transformer) weights** (`rdt-170m` / `rdt-1b`) and write JSON with **`metrics: null`** until you integrate the [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) policy forward pass and a mapping to 8 discrete commands.

- **`saycan`:** Writes a **placeholder** JSON (`metrics: null`). Full [SayCan](https://say-can.github.io/) is an LLM + affordance + low-level policy stack, not one frozen image→class model on SData.

`aggregate_results.py` shows **—** where `metrics` is missing, so these rows still appear in the Markdown table without fake numbers.

**What to install for real RDT / SayCan prediction** (outside this repo’s default env): see [`EXTERNAL_BASELINES.md`](EXTERNAL_BASELINES.md), [`requirements-rdt.txt`](requirements-rdt.txt), [`requirements-saycan.txt`](requirements-saycan.txt), and `python sdata_vla_benchmark/scripts/check_external_baselines.py`.

## Recommended reporting

- **Vision-aligned:** `mmfuse_vision_only` vs `clip_rt` / `openclip_*` (with `SDATA_FROZEN_DUAL_CAM=1`).
- **Best system:** `mmfuse_full` as your full model with audio, clearly **not** comparable to frozen rows on inputs alone.
