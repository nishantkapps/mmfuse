# Text Encoder and Pass-Through (No Text)

This doc describes where a **separate text encoder** for direct text input is used (distinct from audio), and how "no text" is handled (pass-through).

## Separate text vs audio

- **Audio**: Encoded by the audio encoder (Wav2Vec etc.) from microphone/speech. Stored and passed under the `"audio"` modality (768-dim).
- **Text**: Encoded by the text encoder (CLIP/BERT/VisCoP) from **direct text input** (questions, commands, captions). Stored and passed under the `"text"` modality (768-dim).

The fusion accepts both `audio` and `text`. When a pipeline has no direct text (e.g. SData), `"text"` is passed as zeros. When it has no audio (e.g. QA precompute), `"audio"` is passed as zeros.

## Where text is used

| Component | Role | When no text |
|-----------|------|----------------|
| **Finetune** (`training/finetune.py`) | QA datasets have question/options. Text encoder encodes them; fusion gets vision + text. | Dataset has no `question`/`text`: `has_text` is False, **no text encoder**, model is vision-only (pass-through). |
| **Precompute QA** (`experiments/precompute_video_text.py`) | Saves vision + **text** in `"text"` key (768-dim); **audio** key is zeros for QA. | If text missing/empty: **zero vector** in `"text"` (pass-through). |
| **Run dataset** (`experiments/run_dataset.py`) | Loads precomputed `"audio"` and `"text"`; fusion gets both. | Old .pt without `"text"`: default zeros (backward compat). |
| **Robotic feedback system** (`run/robotic_feedback_system.py`) | Text encoder for command text. | If `command_texts` is None or empty: **zero embedding** (pass-through). |
| **SData pipeline** | Vision + **audio** (speech); **no direct text**. | **text** is always zeros (768-dim) so fusion receives both modalities. |

## Design rule

- **With text**: Encode text and feed to fusion (or save in precompute).
- **Without text**: Do not require text. Either (a) omit the text modality (finetune: vision-only model), or (b) pass **zeros** with the same dimension so the fusion forward still receives all keys it was built with.

## Finetune usage

- QA datasets: use `--text-encoder clip` (default) or `bert`; dataset must have `question`/`text` in annotations so `has_text` is True.
- Video-only datasets (e.g. vima_bench with no QA fields): no text in annotations → vision-only model, no text encoder (pass-through).
