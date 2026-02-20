# Cross-Dataset Evaluation: Why Linear Probe?

## The Issue

**VisCoP** achieves 60%+ on VideoMME, NeXTQA, EgoSchema because it is a full VLM: it takes video+text and predicts answers with its own trained head.

**MMFuse** uses VisCoP only as a **feature extractor**. The fusion + classifier were trained on **SData** (8 massage instruction classes). When we apply the SData-trained classifier directly to VideoMME/NeXTQA/Charades:

- **Domain mismatch**: The classifier learned "Start the Massage", "Focus Here", etc. Video QA has different semantics.
- **Charades (157 classes)**: The classifier has 8 outputs. We predict 0–7 for a 157-way problem → ~0% accuracy.
- **VideoMME/NeXTQA**: We slice to 4–5 outputs, but the head was never trained on those tasks → low accuracy.

## Solution: Linear Probe

For cross-dataset benchmarks, we use **linear probe** evaluation by default:

1. Compute fused embeddings (vision + text via MMFuse fusion).
2. Train a linear classifier on the **train split** (80%).
3. Evaluate on the **test split** (20%).

This measures **representation transfer**: how well do the fused embeddings capture task-relevant information? Results can approach VisCoP-level transfer when the representations are good.

## Usage

```bash
# Cross-dataset: linear-probe by default (no flag needed)
python experiments/run_dataset.py --dataset video_mme --checkpoint path/to/model.pt

# SData: zero-shot (trained head)
python experiments/run_dataset.py --dataset sdata --checkpoint path/to/model.pt

# Force zero-shot on cross-dataset (will give low accuracy)
python experiments/run_dataset.py --dataset video_mme --checkpoint path/to/model.pt --zero-shot
```

## L1/L2/L3 and A1–A8 Columns

- **L1, L2, L3**: VIMA-Bench levels. Need `embeddings/vima_bench/` and run evaluation.
- **A1–A8**: SData per-class accuracy. Need `embeddings/sdata_viscop/` and run SData evaluation.

Run the full pipeline to populate all columns:

```bash
./experiments/run_all_cross_dataset_experiments.sh
```
