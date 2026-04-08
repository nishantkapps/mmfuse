# Alternative frozen VLAs (Hugging Face)

All use the **same** eval: `AutoModelForVision2Seq` + `generate` on one camera frame + prompt, then **post-hoc** match of decoded text to the eight SData commands. **No training on SData.**

## Default slot (`openvla` in `run_frozen_benchmarks.py`)

- `openvla/openvla-7b` — default

## Swap via environment or CLI

```bash
export HF_VLA_MODEL_ID="org/model-name"
export HF_VLA_REPORT_AS="my_baseline_name"   # optional: how it appears in JSON/table

python sdata_vla_benchmark/run_frozen_benchmarks.py \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --mmfuse-checkpoint /path/to.pt \
  --openvla-model-id "$HF_VLA_MODEL_ID" \
  --hf-vla-report-as "$HF_VLA_REPORT_AS" \
  --device cuda
```

Or run the generic script:

```bash
python sdata_vla_benchmark/frozen/hf_vla_infer.py \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --output sdata_vla_benchmark/outputs/my_vla.json \
  --model-id org/model-name \
  --report-as my_vla \
  --device cuda
```

## Requirements

The checkpoint must implement **`transformers.AutoModelForVision2Seq`** with **`processor(images=..., text=..., return_tensors="pt")`** and **`model.generate`**. If a model uses a different API (e.g. pure causal LM, JAX-only), it will not work with this harness without a small adapter.

## Examples to try (verify on HF that the repo supports Vision2Seq)

Search [Hugging Face](https://huggingface.co/models?search=vla) for “vla”, “openvla”, “robotics”. Examples that *may* work if they follow the OpenVLA-style API:

- Other OpenVLA checkpoints from the same org (e.g. fine-tuned variants), if listed on HF.
- Any community VLA released with `trust_remote_code` + Vision2Seq.

If loading fails, check the model card for the intended `AutoModel` class and open an issue or add a thin adapter in `frozen/` for that class only.
