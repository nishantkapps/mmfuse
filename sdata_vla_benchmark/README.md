# SData VLA benchmark

**Protocol (what is actually compared):** See [`BENCHMARK.md`](BENCHMARK.md) — SData uses **mid-frames** from each video (not full temporal modeling); frozen CLIP baselines default to **averaging cam1+cam2** to align with MMFuse’s two-view eval.

**RDT / SayCan setup (deps, clones, why `metrics` may be null):** [`EXTERNAL_BASELINES.md`](EXTERNAL_BASELINES.md) · [`requirements-rdt.txt`](requirements-rdt.txt) · [`requirements-saycan.txt`](requirements-saycan.txt) · checklist: `python sdata_vla_benchmark/scripts/check_external_baselines.py`

**Goal:** Compare **your MMFuse checkpoint** to **other frozen models** on **the same SData test split**, with **no training** on those baselines.

Same split as training: [`training/train_sdata_attention.py`](../training/train_sdata_attention.py) (90/10 stratified by `(cam1, cam2)` pair, `random_state=42`; test uses augmentation `v=0` only).

**Default comparison (4 JSON outputs with 8-class metrics):** MMFuse vision-only, MMFuse full, **CLIP-RT**, **OpenCLIP** B-32 (`openclip_b32`).  
Add **`rt1`**, **`rt2`**, **`saycan`** only if you want extra JSON rows (RDT download / placeholders — **no** 8-class metric); see [`EXTERNAL_BASELINES.md`](EXTERNAL_BASELINES.md). Optional: **`openclip_l14`**, **`openvla`**, **`octo`**.

**Why not six different “named” VLAs (OpenVLA, Octo, VIMA, …) in one table?** Each checkpoint expects its **own** action space, tokenizer, and stack. This repo standardizes on **one** protocol: discrete pick among the eight SData command strings. We ship **CLIP-RT** (robotics-tuned contrastive), **OpenCLIP** (generic frozen VL), and optional **HF Vision2Seq** (`openvla` slot) when your environment can load them; wiring six unrelated VLAs end-to-end would be separate adapters per model. See [`ALTERNATIVE_VLAS.md`](ALTERNATIVE_VLAS.md).

## Principles

- **VLAs:** Load **public / frozen** checkpoints — **no fine-tuning on SData**. Run each model’s **normal** inference (`generate`, `sample_actions`, etc.).
- **Metrics:** For text-generating models (e.g. OpenVLA), **post-hoc** string matching maps decoded text to one of the eight SData **command phrases** (from [`config/sdata_movement_config.yaml`](../config/sdata_movement_config.yaml)) — this is **not** a trainable 8-way head.
- **MMFuse:** Your trained checkpoint; vision-only vs full multimodal as two rows.

## Setup

From the **mmfuse repository root**:

```bash
pip install -e .
pip install -r sdata_vla_benchmark/requirements.txt
export PYTHONPATH="/path/to/mmfuse/repo"   # if not using editable install
```

| Variable | Meaning |
|----------|---------|
| `SDATA_ROOT` | Path to `dataset/sdata` |
| `MMFUSE_CHECKPOINT` | Your `.pt` checkpoint |
| `OPEN_VLA_MODEL_ID` / `HF_VLA_MODEL_ID` | Any HF **Vision2Seq** VLA id (default `openvla/openvla-7b`) — swap without code changes |
| `HF_VLA_REPORT_AS` | Short name for JSON/table when using a non-OpenVLA id (optional) |
| `OCTO_PRETRAINED` | e.g. `hf://rail-berkeley/octo-base` |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Seconds for large HF downloads (hub default is short; use **600** on HPC if timeouts) |
| `CLIP_RT_LOCAL_WEIGHTS` | Absolute path to `cliprt-oxe-pretrained.pt` on disk — **skips** hub download |
| `CLIP_RT_FORCE_DOWNLOAD` | Set to `1` to re-download from HF and ignore a **bad/partial** cached file (~11.8 GB) |
| `OPENCLIP_ZS_TEXT_TEMPLATE` | e.g. `{}` or `massage instruction: {}` — wraps each SData command for OpenCLIP text encoding (default `{}`) |
| `SDATA_FROZEN_DUAL_CAM` | `1` (default): average frozen CLIP/CLIP-RT/OpenCLIP scores over **cam1+cam2** mid-frames; `0`: single-camera eval |

### OpenVLA fails: TIMM / `transformers` / `_attn_implementation`

1. **`timm`:** OpenVLA needs **timm below 1.0** (e.g. `0.9.x`).

2. **`transformers` too old (e.g. 4.35):** OpenVLA’s remote code targets **`transformers` 4.40+**. Without upgrading, you may see  
   `'OpenVLAConfig' object has no attribute '_attn_implementation'`.  
   **`hf_vla_core` patches missing `_attn_implementation` on the loaded config** (default `eager`) so frozen eval can run on older shared environments. For full parity with upstream OpenVLA, prefer [`requirements-openvla.txt`](requirements-openvla.txt). Optional override: `HF_VLA_ATTN_IMPLEMENTATION=sdpa` when your stack supports it.

```bash
pip install -r sdata_vla_benchmark/requirements-openvla.txt
```

### HPC / slow network

- **CLIP-RT / large `.pt` files:** Default HF read timeout is **10s**, which often fails. Either `export HF_HUB_DOWNLOAD_TIMEOUT=600` before running, or download `cliprt-oxe-pretrained.pt` once (browser or `huggingface-cli download`) and set `CLIP_RT_LOCAL_WEIGHTS=/path/to/cliprt-oxe-pretrained.pt`.
- **Corrupt partial download:** If you see `Consistency check failed: file should be of size …`, run again with `CLIP_RT_FORCE_DOWNLOAD=1` or `python .../clip_rt_infer.py ... --force-download`. The script also **auto-retries** once with a forced re-download. If it still fails, remove the broken blob under `~/.cache/huggingface/hub/models--clip-rt--clip-rt-oxe-pretrained/` and retry.
- **OpenVLA:** Code uses `low_cpu_mem_usage=False` so **Accelerate** is not required for loading.
- **Octo:** Needs **JAX** + the `octo` package; omit `octo` from `--models` if not installed.

## 1. Build the manifest

```bash
python sdata_vla_benchmark/scripts/build_manifest.py \
  --dataset "${SDATA_ROOT:-dataset/sdata}" \
  --out sdata_vla_benchmark/manifests/sdata_manifest.csv
```

## 2. Run the full frozen comparison (recommended default)

```bash
mkdir -p sdata_vla_benchmark/outputs
export HF_HUB_DOWNLOAD_TIMEOUT=600

python sdata_vla_benchmark/run_frozen_benchmarks.py \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --split test \
  --output-dir sdata_vla_benchmark/outputs \
  --mmfuse-checkpoint "${MMFUSE_CHECKPOINT}" \
  --device cuda
```

This runs the **default four** keys (`mmfuse_*`, `clip_rt`, `openclip_b32`).

**Optional extra keys** (not in default — **no** 8-class metric in this repo): **`rt1`**, **`rt2`** (RDT HF snapshot), **`saycan`** (placeholder). See [`EXTERNAL_BASELINES.md`](EXTERNAL_BASELINES.md). Append e.g. **`openclip_l14`**, **`openvla`**, **`octo`** to `--models` as needed.

```bash
python sdata_vla_benchmark/run_frozen_benchmarks.py \
  ... --models mmfuse_vision_only mmfuse_full clip_rt openclip_b32 openclip_l14 openvla
```

- **OpenCLIP ZS:** [`frozen/openclip_zs_infer.py`](frozen/openclip_zs_infer.py) — LAION/OpenCLIP weights; softmax over image–text similarity for the eight commands (presets `b32`, `l14`, `b16`). **First run** downloads large checkpoints from the hub; the script sets `HF_HUB_DOWNLOAD_TIMEOUT` to **600s** if unset (same issue as CLIP-RT if you see read timeout).
- **OpenVLA:** [`frozen/openvla_infer.py`](frozen/openvla_infer.py) — HF `generate()` + post-hoc label match on decoded text.
- **Octo:** [`frozen/octo_infer.py`](frozen/octo_infer.py) — `sample_actions` if `octo` + JAX are installed; outputs may be **continuous** (metrics `null`).
- **CLIP-RT:** [`frozen/clip_rt_infer.py`](frozen/clip_rt_infer.py) — downloads `clip-rt/clip-rt-oxe-pretrained` from HF, runs **native** open_clip contrastive scoring over the eight SData command strings (paper-style primitive pick).
- **RT-1 / RT-2 (`rt1`, `rt2`):** [`frozen/rt_infer.py`](frozen/rt_infer.py) → [`frozen/rdt_infer.py`](frozen/rdt_infer.py): download **RDT** on HF (`rdt-170m` / `rdt-1b`). **Not** Google’s original RT weights. **`metrics`** stay `null` until you wire upstream **diffusion** inference (`RDT_ROOT`).
- **SayCan (`saycan`):** [`frozen/saycan_infer.py`](frozen/saycan_infer.py) — table row + manifest alignment only; **`metrics: null`** (multi-component system, not one checkpoint).

## 3. Run models individually

**MMFuse**

```bash
python sdata_vla_benchmark/mmfuse_eval/run_eval.py \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --checkpoint "${MMFUSE_CHECKPOINT}" \
  --mode vision_only \
  --output sdata_vla_benchmark/outputs/mmfuse_vision_only.json
```

**OpenVLA only**

```bash
python sdata_vla_benchmark/frozen/openvla_infer.py \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --output sdata_vla_benchmark/outputs/openvla.json
```

## 4. Aggregate a table

```bash
python sdata_vla_benchmark/aggregate_results.py \
  --inputs sdata_vla_benchmark/outputs/*.json \
  --markdown sdata_vla_benchmark/outputs/table.md \
  --show
```

## Layout

```
sdata_vla_benchmark/
  BENCHMARK.md                 # eval protocol (mid-frame, dual-cam, audio)
  run_frozen_benchmarks.py    # orchestrator
  frozen/
    common.py                  # manifest, frames, post-hoc text→label
    openvla_infer.py           # frozen OpenVLA (HF)
    octo_infer.py              # frozen Octo (JAX) — continuous output possible
    rt_infer.py                # RT-1/2 → RDT hub download + notes
    rdt_infer.py               # snapshot_download RDT weights from HF
    clip_rt_infer.py           # CLIP-RT (HF weights + open_clip)
    openclip_zs_infer.py       # zero-shot OpenCLIP (LAION presets)
    rdt_infer.py               # RDT HF download (rt1/rt2 proxy)
    saycan_infer.py            # SayCan placeholder JSON (no single-model eval)
  scripts/build_manifest.py
  mmfuse_eval/run_eval.py
  metrics/classification.py
  manifests/
  outputs/                     # gitignored
```
