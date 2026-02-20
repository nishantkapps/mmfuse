# Ablation Experiments for MMFuse SData

This document describes the ablation experiments for the paper and how to run them.

## Experiment Table (Template)

| Experiment | Description | Test Acc. (%) |
|------------|-------------|---------------|
| full | Full model: vision (both cameras) + audio + pressure + emg | — |
| no_vision_cam1 | Ablation: remove vision camera 1 (zero out) | — |
| no_vision_cam2 | Ablation: remove vision camera 2 (zero out) | — |
| no_audio | Ablation: remove audio (zero out) | — |
| vision_only | Vision only (both cameras) | — |
| vision_cam1_only | Vision camera 1 only | — |
| vision_cam2_only | Vision camera 2 only | — |
| audio_only | Audio only | — |
| fusion_concat | Replace attention fusion with concat-project | — |
| no_kl_loss | Full model without KL divergence loss | — |
| no_movement_head | Full model without movement prediction head | — |

*Fill in Test Acc. after running experiments.*

## How to Run

### 1. Precompute embeddings (if not done)

```bash
python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata --out-dir embeddings/sdata_viscop \
    --vision-encoder viscop --audio-encoder wav2vec --cross-pair
```

### 2. Run a single ablation

Use `--dataset` for correct train/test split (split by video pairs before augmentation):

```bash
python scripts/run_ablation.py --ablation no_audio --embeddings-dir embeddings/sdata_viscop --dataset dataset/sdata --epochs 10
```

Results are saved to `ablation_runs/no_audio/`.

### 3. Run all ablations

```bash
EMBEDDINGS_DIR=embeddings/sdata_viscop DATASET=dataset/sdata EPOCHS=10 bash scripts/run_all_ablations.sh
```

Or manually:
```bash
for ablation in full no_vision_cam1 no_vision_cam2 no_audio vision_only vision_cam1_only vision_cam2_only audio_only fusion_concat no_kl_loss no_movement_head; do
  python scripts/run_ablation.py --ablation $ablation --embeddings-dir embeddings/sdata_viscop --dataset dataset/sdata --epochs 10
done
```

### 4. Collect results (table + figures)

```bash
python scripts/collect_ablation_results.py --ablation-dir ablation_runs --output-dir results
```

Generates:
- `results/ablation_table.tex` – LaTeX table (Experiment | Test Acc. | Δ from Full)
- `results/ablation_table.csv` – CSV
- `results/ablation_fig_accuracy.png` – Bar chart of test accuracy per experiment
- `results/ablation_fig_delta.png` – Bar chart of Δ from full (impact of each ablation)
- `results/ablation_fig_scatter.png` – Scatter plot of Δtrain vs Δtest (when 2+ ablations)

## Experiment Definitions

- **full**: Baseline with all modalities (vision cam1, cam2, audio, pressure, emg) and all components (attention fusion, KL loss, movement head).
- **no_X**: Leave-one-out; zero out modality X while keeping others.
- **X_only**: Single-modality baseline; only modality X is used.
- **fusion_concat**: Use `MultimodalFusion` (concat-project) instead of `MultimodalFusionWithAttention`.
- **no_kl_loss**: Train without KL divergence between modalities.
- **no_movement_head**: Train without the movement prediction auxiliary task.

## Adding New Experiments

Edit `config/ablation_experiments.yaml` to add new ablation configs. Each config specifies:

- `use_vision_cam1`, `use_vision_cam2`, `use_audio`, `use_pressure`, `use_emg`: which modalities to use (false = zero out)
- `fusion_type`: `"attention"` or `"concat"`
- `use_kl_loss`: whether to use KL divergence loss
- `use_movement_head`: whether to use movement prediction head
