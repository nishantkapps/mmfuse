# Dataset Download Guide (Subsets for MMFuse Experiments)

Download subsets of each benchmark into `extdataset/<name>/`. Use subsets for faster experiments; full datasets for final evaluation.

---

## 1. VideoMME (subset) – Exocentric Video QA

**Source:** [HuggingFace lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)  
**Full:** 2,700 QA pairs, 900 videos (~101 GB with videos)  
**Subset:** Use first N samples from parquet for quick eval.

```bash
mkdir -p extdataset/video_mme
cd extdataset/video_mme
```

**Option A – HuggingFace (annotations only, ~405 KB):**
```bash
pip install huggingface_hub
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/Video-MME', split='test')
# Subset: first 100 samples
ds_sub = ds.select(range(min(100, len(ds))))
ds_sub.to_parquet('video_mme_subset.parquet')
"
```

**Option B – Full data (parquet + video zips):**  
Download from [Video-MME website](https://video-mme.github.io/) or HuggingFace. Place:
- `*.parquet` – annotations
- `videos_chunked_*.zip` – video files

Then run:
```bash
python experiments/prepare_video_mme_from_parquet.py --data-dir extdataset/video_mme --extract-zips
```

**Expected layout:** `extdataset/video_mme/annotations.json`, `videos/*.mp4`

---

## 2. NeXTQA (subset) – Exocentric Video QA

**Source:** [HuggingFace lmms-lab/NExTQA](https://huggingface.co/datasets/lmms-lab/NExTQA)  
**Full:** 60,608 QA pairs, 5,440 videos  
**Subset:** Use val/test split or first N samples.

```bash
mkdir -p extdataset/nextqa
cd extdataset/nextqa
```

**Download subset:**
```bash
pip install huggingface_hub
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/NExTQA', 'MC', split='test')  # MC = multiple choice
# Subset: first 200 samples
ds_sub = ds.select(range(min(200, len(ds))))
# Convert to our format: video_id, question, a0..a4, answer
rows = []
for i, row in enumerate(ds_sub):
    rows.append({
        'video_id': row.get('video_id', str(i)),
        'question': row.get('question', ''),
        'a0': row.get('a0', ''),
        'a1': row.get('a1', ''),
        'a2': row.get('a2', ''),
        'a3': row.get('a3', ''),
        'a4': row.get('a4', ''),
        'answer': int(row.get('answer', 0))
    })
import json
with open('annotations.json', 'w') as f:
    json.dump(rows, f, indent=2)
print('Saved annotations.json. Download videos separately from NExT-QA repo.')
"
```

**Videos:** Download from [NExT-QA GitHub](https://github.com/doc-doc/NExT-QA) (requires form/agreement). Place in `extdataset/nextqa/videos/`.

**Expected layout:** `extdataset/nextqa/annotations.json`, `videos/<id>.mp4`

---

## 3. Charades – Action Recognition

**Source:** [HuggingFace Aditya02/Charades-Action-Sequence-Sample](https://huggingface.co/datasets/Aditya02/Charades-Action-Sequence-Sample) or [Allen AI Charades](https://prior.allenai.org/projects/charades)  
**Full:** 1,000 samples (subset); original Charades has 9,848 videos, 157 action classes  
**Subset:** Use first N videos for quick eval.

```bash
mkdir -p extdataset/charades
cd extdataset/charades
```

**Option A – HuggingFace (recommended):**
```bash
python experiments/download_dataset_subsets.py charades --max-samples 100
# Saves extdataset/charades/annotations.json
# Videos are downloaded by HuggingFace to cache. Copy to extdataset/charades/videos/ if needed.
```

Or manually:
```bash
pip install datasets
python -c "
from datasets import load_dataset
import json
ds = load_dataset('Aditya02/Charades-Action-Sequence-Sample', split='train')
rows = []
for i, row in enumerate(ds):
    if i >= 100: break
    vid = row.get('video_id', str(i))
    labels = row.get('labels', [])
    labels = eval(labels) if isinstance(labels, str) else (labels or [])
    rows.append({
        'video_id': vid,
        'video_path': f'videos/{vid}.mp4',
        'text': row.get('script', ''),
        'target': int(labels[0]) if labels else 0
    })
with open('annotations.json', 'w') as f:
    json.dump(rows, f, indent=2)
"
```

**Option B – Original Charades:** Download from [Allen AI](https://prior.allenai.org/projects/charades). Use `Charades_v1_train.csv`, `Charades_v1_test.csv`, and `Charades_v1_480/` videos.

**Expected layout:** `extdataset/charades/annotations.json`, `videos/*.mp4`

---

## 4. EgoSchema (subset) – Egocentric Video QA

**Source:** [EgoSchema GitHub](https://github.com/egoschema/EgoSchema) | [HuggingFace lmms-lab/egoschema](https://huggingface.co/datasets/lmms-lab/egoschema) | [Kaggle](https://www.kaggle.com/competitions/egoschema-public)  
**Full:** 5,000+ QA pairs, 250+ hours  
**Subset:** 500 questions with public answers (for offline eval).

**Recommended – Download script (HuggingFace):**
```bash
python experiments/download_egoschema.py --max-samples 500 --video-zips 2
# Or annotations only (no videos): --skip-videos
```

Or via `download_datasets_full.py`:
```bash
python experiments/download_datasets_full.py egoschema --max-samples 500 --video-zips 2
```

**Alternative – Kaggle:**
1. Accept terms at [egoschema-public](https://www.kaggle.com/competitions/egoschema-public)
2. `kaggle competitions download -c egoschema-public` then unzip

**Expected layout:** `extdataset/egoschema/annotations.json`, `videos/*.mp4`

---

## 5. VIMA-Bench – Robotic Control (Simulator Eval)

**Source:** [VIMA-Bench GitHub](https://github.com/vimalabs/VIMABench) | [HuggingFace VIMA/VIMA-Data](https://huggingface.co/datasets/VIMA/VIMA-Data)  
**Note:** VIMA-Bench is simulator-based. For embedding-based eval, use pre-recorded trajectories or export frames.

```bash
mkdir -p extdataset/vima_bench
cd extdataset/vima_bench
```

**Option A – Simulator (recommended):** Clone and run the VIMA-Bench simulator for L1/L2/L3 evaluation. See [VIMABench](https://github.com/vimalabs/VIMABench).

**Option B – Precomputed frames:** Download VIMA-Data from HuggingFace. Create `annotations.json` with `video_path`, `text` (task prompt), `target` (0=L1, 1=L2, 2=L3).

```bash
pip install huggingface_hub
# Download VIMA-Data and extract trajectories; map to L1/L2/L3 by task type
```

**Expected layout:** `extdataset/vima_bench/annotations.json` with `target`: 0=L1, 1=L2, 2=L3.

---

## 6. SData – Your Main Benchmark

**Source:** Local `dataset/sdata/` (already in project)  
No download needed. Use:
```bash
python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata \
    --out-dir embeddings/sdata_viscop --vision-encoder viscop --audio-encoder wav2vec --cross-pair
```

---

## Quick Reference: Target Layout

| Dataset    | Dir                | Key files                          |
|-----------|--------------------|------------------------------------|
| VideoMME  | extdataset/video_mme  | annotations.json, videos/*.mp4    |
| NeXTQA    | extdataset/nextqa    | annotations.json, videos/*.mp4     |
| Charades  | extdataset/charades  | annotations.json or Charades_v1_*.csv, videos/ |
| EgoSchema | extdataset/egoschema | annotations.json, videos/*.mp4     |
| VIMA-Bench| extdataset/vima_bench| annotations.json (target 0/1/2)   |
| SData     | dataset/sdata        | (existing)                         |

---

## After Download: Precompute

```bash
# VideoMME
python experiments/precompute_video_mme.py --out-dir embeddings/video_mme

# NeXTQA
python experiments/precompute_nextqa.py --out-dir embeddings/nextqa

# Charades
python experiments/precompute_charades.py --out-dir embeddings/charades

# EgoSchema
python experiments/precompute_egoschema.py --out-dir embeddings/egoschema

# VIMA-Bench (custom precompute with L1/L2/L3 targets)
python experiments/precompute_video_text.py --dataset vima_bench --out-dir embeddings/vima_bench
```
