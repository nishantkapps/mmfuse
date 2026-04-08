# RT-1 / RT-2 (via RDT) and SayCan — what you need for *prediction*

This repo’s **`rt1` / `rt2`** keys download **[RDT](https://huggingface.co/robotics-diffusion-transformer)** weights (`rdt-170m` / `rdt-1b`). **Google’s original RT-1/RT-2 checkpoints are not public** as a single Hugging Face `transformers` model.

**`saycan`** is the [SayCan](https://say-can.github.io/) system (LLM + affordances + sim policies), **not** one `.pt` file.

---

## RDT — running real `policy` forward (optional, outside mmfuse)

1. **Separate conda env** (RDT pins `timm==1.0`, `transformers==4.41`, etc. — conflicts with some mmfuse stacks).

2. **Clone and install** (from [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer)):
   ```bash
   git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git
   cd RoboticsDiffusionTransformer
   conda create -n rdt python=3.10 -y
   conda activate rdt
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
   pip install packaging==24.0
   pip install flash-attn --no-build-isolation   # may fail on CPU-only machines
   pip install -r requirements.txt
   ```

3. **Weights:** Already downloadable via Hugging Face:
   - `robotics-diffusion-transformer/rdt-170m`
   - `robotics-diffusion-transformer/rdt-1b`  
   Or point `create_model` / `RDTRunner.from_pretrained` at the snapshot path (see upstream `scripts/agilex_model.py`).

4. **Encoders:** `scripts/agilex_model.py` uses **`google/siglip-so400m-patch14-384`** from HF for vision. Language is usually **precomputed embeddings** (see upstream `scripts/encode_lang.py`); the deployed stack often **does not** load full T5-XXL on GPU at once.

5. **Inference API:** `RoboticDiffusionTransformerModel.step(proprio, images, text_embeds)` expects:
   - **Proprio:** bimanual joint state (14-D for AgileX-style).
   - **Images:** 6 images in order (2 time steps × 3 cameras) — see upstream `agilex_inference.py` (ROS-driven in the official script).
   - **Text embeds:** instruction embeddings, not raw strings only.

6. **SData gap:** RDT outputs **continuous robot actions** in a **unified action vector**, not one of your **eight massage command IDs**. Turning that into **8-class accuracy** requires either:
   - fine-tuning / calibration on SData, or  
   - a hand-designed mapping (generally **not** meaningful across domains).

**Conclusion:** Use RDT in the **official repo + sim/real stack** for manipulation metrics; treat SData **8-way accuracy** as **undefined** for RDT until you define a task-specific mapping.

Pinned mirrors of upstream Python deps: [`requirements-rdt.txt`](requirements-rdt.txt).

---

## SayCan — what “full prediction” needs

From **`google-research/google-research/saycan`** (notebook `SayCan-Robot-Pick-Place.ipynb`):

1. **OpenAI API key** (GPT-3 in the paper; you can try newer models with code changes).
2. **Python packages** (see [`requirements-saycan.txt`](requirements-saycan.txt)): **CLIP** (`pip install git+https://github.com/openai/CLIP.git`), **Flax/Jax**, **PyBullet**, **TensorFlow** (ViLD / policies), **gdown** for assets.
3. **Weights / assets:** Notebook uses **gdown** IDs for UR5, gripper, bowls, **ViLD** checkpoints (`gsutil` paths in Google’s instructions), and **low-level policies** trained for their tabletop tasks.
4. **Environment:** PyBullet **tabletop** scene — **not** your massage RGB clips fed into the same pipeline without a full reimplementation.

**Conclusion:** SayCan is **not** “install one pip package and run on `sdata_manifest.csv`.” The `saycan` JSON in this benchmark is a **placeholder row** until you port or replace the pipeline.

---

## Env vars (this repo)

| Variable | Purpose |
|----------|---------|
| `RDT_ROOT` | Path to a clone of `RoboticsDiffusionTransformer` (for your own scripts / future wiring). |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Increase (e.g. `600`) when downloading RDT snapshots. |

---

## Checklist helper

From repo root:

```bash
python sdata_vla_benchmark/scripts/check_external_baselines.py
```

Prints whether optional imports/paths are visible (does not download gigabyte models).
