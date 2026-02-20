# Paper Table Guide: VisCoP vs MMFuse

## What VisCoP Table 3 Shows (Adaptation Strategies)

The VisCoP paper Table 3 compares **adaptation strategies** for robotic control:

| Component | Role |
|-----------|------|
| **VL-C** (Visual-Language Conditioning) | Conditions the model on task instructions |
| **VE** (Visual Embedding) | Adapts visual encoder for robotic domain |
| **LLM** | Language model for action prediction |

Each row is a different **strategy** (which components are enabled). They evaluate:
- **Target (Robotic)**: VIMA-Bench L1, L2, L3
- **Source (Human Understanding)**: Ego-in-Exo, NeXTQA, VideoMME, ADL-X MCQ, ADL-X Desc

The table shows the **adaptation tradeoff**: improving robotic control (target) can hurt human video understanding (source).

---

## What MMFuse Does (Different Story)

**MMFuse is NOT doing VisCoP-style adaptation.** Our setup:

| Component | MMFuse |
|-----------|--------|
| Vision | VisCoP (frozen backbone) |
| Fusion | Our MultimodalFusionWithAttention (trained on SData) |
| Head | ActionClassifier (8 classes) + optional MovementHead |
| Training | SData (massage instruction following) |

We use VisCoP as a **frozen feature extractor**. We do not adapt it for robotics. Our contribution is:
- **Multimodal fusion** for HRI (vision + audio + pressure + EMG)
- **SData benchmark** (massage instruction following)
- **Movement head** (predicts delta for robot control)

---

## Our Table Columns: What They Mean

| Column | Source | How to Populate |
|--------|--------|-----------------|
| **L1, L2, L3** | VIMA-Bench (robotic simulation) | Run VIMA-Bench evaluation. Requires simulator or pre-recorded data. |
| **A1–A8** | SData per-class accuracy | Run SData evaluation. Requires `embeddings/sdata_viscop/`. |
| **EgoSchema, NeXTQA, VideoMME, ADL-X** | Video QA benchmarks | Linear-probe on our fused embeddings (transfer evaluation). |
| **Avg** | Average of video QA accuracies | Computed from EgoSchema, NeXTQA, VideoMME, ADL-X. |

**To get A1–A8**: Run the full pipeline. SData is now included (step 4). Ensure you have:
```bash
python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata \
  --out-dir embeddings/sdata_viscop --vision-encoder viscop --audio-encoder wav2vec --cross-pair
```

**To get L1/L2/L3**: VIMA-Bench requires the simulator. See `DATASET_DOWNLOAD.md`. If unavailable, these columns stay "---".

---

## Paper-Worthy Narrative

### Option A: SData-Centric (Recommended)

**Primary contribution**: Multimodal fusion for human-robot interaction (massage instruction following).

**Table structure**:
- **Primary**: SData (A1–A8, overall accuracy, movement MSE)
- **Transfer**: Video QA (NeXTQA, VideoMME, ADL-X) via linear probe – shows representation quality
- **Optional**: VIMA-Bench (L1/L2/L3) if you add it – would show transfer to robotic simulation

**Narrative**: "We train MMFuse on SData for massage instruction following. Using VisCoP as a frozen vision encoder, our fusion model learns to map vision+audio to 8 actions and predicts movement deltas. We evaluate on SData (primary) and probe transfer to video QA benchmarks to assess representation quality."

### Option B: Align with VisCoP Format (If You Add VIMA-Bench)

If you run VIMA-Bench, your table can mirror VisCoP's structure:
- **Robotic (target)**: L1, L2, L3
- **Human understanding (source)**: NeXTQA, VideoMME, ADL-X

But note: We are not adapting VisCoP. We are using it as a backbone and training our own fusion. The comparison would be "MMFuse (our fusion + SData head) vs VisCoP (full VLM)" – different model classes.

### Option C: Add VisCoP Baseline Row

For a stronger paper, add a **VisCoP baseline** row (from the paper: NeXTQA 83.71, VideoMME 63.67, ADL-X ~55–66). This shows:
- VisCoP (direct): full VLM evaluation
- MMFuse (linear probe): our fused representations

Our numbers will typically be lower (linear probe vs full model) but the gap indicates how much task-specific tuning helps.

---

## Summary

1. **L1/L2/L3**: Need VIMA-Bench. Add when you have simulator data.
2. **A1–A8**: Need SData. Now in pipeline – run `./experiments/run_all_cross_dataset_experiments.sh` with `embeddings/sdata_viscop/` present.
3. **We are not doing adaptation** like VisCoP. Our story is multimodal fusion for HRI + SData benchmark + movement head.
4. **Video QA columns**: Linear-probe measures transfer. For paper, consider adding VisCoP baseline row for comparison.
