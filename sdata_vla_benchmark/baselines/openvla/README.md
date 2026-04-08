# OpenVLA (frozen)

Use the official OpenVLA checkpoint and inference API. Do **not** attach MMFuse modules.

## Suggested workflow

1. Install dependencies from the [OpenVLA repository](https://github.com/openvla/openvla) (transformers, etc.).
2. For **action-class** evaluation on SData, feed the same frames as other baselines (from `manifests/sdata_manifest.csv`: `cam1`, `cam2` paths) plus a task prompt listing the eight massage commands.
3. Map the model’s output (generated text or discrete action) to class indices `0..7` using constrained decoding or string matching; write JSON in the schema below.

## Output JSON schema

```json
{
  "model": "openvla",
  "checkpoint": "hf:openvla/openvla-7b",
  "split": "test",
  "manifest": "path/to/sdata_manifest.csv",
  "metrics": { "accuracy": 0.0, "macro_f1": 0.0, "n": 0 },
  "predictions": [
    { "sample_id": 0, "split": "test", "label": 3, "pred_idx": 3, "raw": "..." }
  ]
}
```

Implement `run_openvla.py` locally when dependencies are available; keep it in this folder so OpenVLA/JAX stacks stay isolated from core training code.
