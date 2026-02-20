# Expected Data Formats for Cross-Dataset Experiments

Place each dataset in `extdataset/<name>/` with the structure below.

## Charades (ADL-X)

```
extdataset/charades/
├── videos/                    # or Charades_v1_480/
│   └── <video_id>.mp4
├── annotations.json           # optional: custom format
└── Charades_v1_train.csv      # or Charades_v1_test.csv
    Charades_v1_test.csv
```

**annotations.json** (optional, overrides CSV):
```json
[
  {"video_path": "videos/abc123.mp4", "text": "description", "target": 0},
  ...
]
```

**Charades CSV** (native): columns `id`, `actions` (e.g. "c001 c005"), `description`. Primary action (first in list) maps to class 0–156.

---

## VideoMME

```
extdataset/video_mme/
├── videos/
│   └── <video_id>.mp4
└── annotations.json
```

**annotations.json**:
```json
[
  {
    "video_id": "vid001",
    "question": "What is happening?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  },
  ...
]
```
Answer maps to target: A→0, B→1, C→2, D→3.

---

## NeXTQA

```
extdataset/nextqa/
├── videos/
│   └── <video_id>.mp4
└── annotations.json           # or nextqa_train.json, nextqa_val.json
```

**annotations.json**:
```json
[
  {
    "video_id": "vid001",
    "question": "Why did X happen?",
    "a0": "option A", "a1": "option B", ...,
    "answer": 0
  },
  ...
]
```
`answer` is 0–4 (MCQ index).

---

## EgoSchema

```
extdataset/egoschema/
├── videos/
│   └── <video_id>.mp4
└── annotations.json
```

**annotations.json**:
```json
[
  {
    "video_id": "vid001",
    "question": "...",
    "options": ["A", "B", "C", "D", "E"],
    "answer": "B"
  },
  ...
]
```
Answer A→0, B→1, ..., E→4.

---

## VIMA-Bench

Simulator-based. Results reported on VisCoP levels: L1 (Object Placement), L2 (Novel Combination), L3 (Novel Object).

**annotations.json** (for precompute):
```json
[
  {"video_path": "videos/task1.mp4", "text": "prompt", "target": 0},
  ...
]
```
`target`: 0=L1, 1=L2, 2=L3. Use precomputed embeddings with config `num_classes: 3`.

---

## Generic (annotations.json / annotations.csv)

For any video+text dataset, create:

```
extdataset/<name>/
├── videos/
│   └── *.mp4
└── annotations.json   # or annotations.csv
```

**annotations.json**:
```json
[
  {"video_path": "videos/x.mp4", "text": "question or caption", "target": 0},
  ...
]
```

**annotations.csv**: columns `video_path`, `text`, `target`.

Then run:
```bash
python experiments/precompute_video_text.py --dataset <name> --out-dir embeddings/<name>
```
