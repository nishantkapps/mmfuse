# How to Show MMFuse vs Other Models

This document describes how to produce **paper-ready comparison tables** with MMFuse vs CLIP, VisCoP, and published baselines on established datasets (NeXTQA, VideoMME, ADL-X) and SData.

---

## Quick Start: Paper Comparison Pipeline

```bash
# 1. Precompute CLIP embeddings (video_mme, nextqa, charades, sdata)
./experiments/run_precompute_clip.sh

# 2. Run full paper comparison (produces CSV + LaTeX)
./experiments/run_all_paper_comparisons.sh
# Or with checkpoint: CHECKPOINT=checkpoints/ckpt_sdata_epoch_5.pt ./experiments/run_all_paper_comparisons.sh
```

**Prerequisites:**
- VisCoP embeddings: `embeddings/video_mme`, `embeddings/nextqa`, `embeddings/charades`, `embeddings/sdata_viscop`
- MMFuse results on cross-dataset: run `./experiments/run_all_cross_dataset_experiments.sh` first to populate `experiments/results/{video_mme,nextqa,charades}/results.json`

**Output:** `experiments/results/paper_comparison.csv`, `paper_comparison.tex`

**Table columns:** NeXTQA | VideoMME | ADL-X | SData | Movement MSE

**Models compared:**
- **Published (cited):** VisCoP (paper), LLaVA-NeXT (paper)
- **Linear probe (our embeddings):** CLIP, VisCoP, MMFuse
- **SData baselines:** VisCoP vision-only, Audio-only (Wav2Vec), VisCoP+Audio (concat), CLIP vision-only
- **MMFuse (full):** Trained fusion on SData + movement head

---

## Three Contributions in Comparison to Other Models

You want to showcase three contributions in comparison to other models:
1. **Multimodal fusion for HRI** (vision + audio + pressure + EMG)
2. **SData benchmark** (massage instruction following)
3. **Movement head** for robot control

---

## 1. Capability Comparison Table (No New Experiments)

Create a **Table: Model capabilities** that shows what each system supports. This is a qualitative comparison—no new runs needed.

| Model | Modalities | SData / HRI | Movement Prediction | Robot Control |
|-------|------------|-------------|---------------------|---------------|
| VisCoP | Vision, Language | No | No | VIMA-Bench (simulation) |
| CLIP / VLMs | Vision, Language | No | No | No |
| Prior HRI (e.g. [cite]) | Varies | Often vision-only or vision+audio | Rare | Sometimes |
| **MMFuse** | **Vision, Audio, Pressure, EMG** | **Yes (SData)** | **Yes** | **Yes (real robot)** |

**Narrative**: "Unlike general-purpose VLMs (VisCoP, CLIP) that use only vision and language, MMFuse fuses four modalities—vision, audio, pressure, and EMG—designed for physical HRI. We introduce the SData benchmark for massage instruction following and a movement head that predicts control deltas for real robot deployment."

---

## 2. SData Benchmark: "First to Evaluate On"

SData is **your benchmark**. Other models have not been evaluated on it. You can state:

- "To our knowledge, MMFuse is the first model evaluated on massage instruction following (SData)."
- "We provide the SData benchmark and baseline results for future comparison."

If you want a **numerical comparison** on SData, you would need to:
- Run a simpler baseline (e.g. vision-only) on SData—your ablations already give this (e.g. `vision_only`, `audio_only`). You can cite those numbers without re-running.
- Or run VisCoP on SData: use VisCoP to encode video+instruction and map to 8 classes. That would require a small evaluation script.

---

## 3. Movement Head: Unique Capability

No standard VLM (VisCoP, CLIP, etc.) predicts movement deltas for robot control. You can:

**A. State it as a unique feature**
- "MMFuse includes a movement head that predicts (Δx, Δy, magnitude) for robot control, which general VLMs do not provide."

**B. Add a simple baseline (optional, minimal code)**
- Baseline: always predict the mean movement (or zero).
- Compare: Movement MSE of MMFuse vs mean/zero baseline.
- This shows the head adds information beyond a trivial predictor.

---

## 4. Suggested Paper Tables

**Table 1: Main results (existing)**  
SData accuracy (A1–A8), video QA transfer (NeXTQA, VideoMME, ADL-X), Movement MSE.

**Table 2: Comparison with related work (new, qualitative)**  
Use the capability table above. Columns: Model, Modalities, SData, Movement Head, Robot Eval.

**Table 3: Ablation (existing)**  
Use your current ablation results. One sentence: "Ablations show that each modality contributes to performance."

---

## 5. Minimal Additions (If You Want One Extra Comparison)

**VisCoP on SData (optional)**  
To get a direct comparison on SData:
- Use VisCoP to encode each (video, instruction) pair.
- Train a linear classifier on VisCoP embeddings → 8 classes.
- Report accuracy vs MMFuse.

That gives: "VisCoP (vision+language, linear probe): X% | MMFuse (vision+audio+pressure+EMG, trained): Y%"

---

## 6. Get Experimental Numbers: Run Baselines on SData

**Script**: `experiments/run_sdata_baselines.py`

This runs simple baselines on SData and compares to MMFuse. Produces numbers for the paper.

```bash
python experiments/run_sdata_baselines.py --checkpoint checkpoints/ckpt_sdata_epoch_N.pt
```

**Baselines** (linear classifier on embeddings):
- **Vision-only**: VisCoP vision (cam1+cam2) → 8 classes
- **Audio-only**: Wav2Vec audio → 8 classes
- **Vision+Audio (concat)**: Concatenate embeddings → 8 classes (no learned fusion)

**MMFuse**: Trained fusion + classifier (vision+audio+pressure+EMG)

**Output**: `experiments/results/sdata_comparison.csv`, `sdata_comparison.tex`

Example table:
| Model | SData Acc (%) | Movement MSE |
|-------|---------------|---------------|
| Vision-only (VisCoP) | X | --- |
| Audio-only (Wav2Vec) | Y | --- |
| Vision+Audio (concat) | Z | --- |
| **MMFuse (full)** | **W** | **M** |

This gives you experimental numbers showing MMFuse excels over baselines on SData.

---

## Summary

| Contribution | How to Show vs Other Models |
|--------------|-----------------------------|
| **Multimodal fusion** | Run `run_sdata_baselines.py` → MMFuse vs Vision-only, Audio-only, Vision+Audio. Capability table. |
| **SData benchmark** | Same script: MMFuse as baseline; first benchmark for massage instruction. |
| **Movement head** | MMFuse row has Movement MSE; baselines have "---". Unique to MMFuse. |
