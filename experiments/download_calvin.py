#!/usr/bin/env python3
"""
Download a small CALVIN subset (Calvin ABC) for manipulation experiments.

This script downloads the tabular CALVIN ABC data from HuggingFace and stores it
locally as a single parquet file. It does NOT download any simulator assets.

Source:
  - HuggingFace dataset: InternRobotics/InternData-Calvin_ABC

Output structure (default):
  extdataset/calvin_abc/
    calvin_abc.parquet      # concatenated subset (max_samples rows)
    meta.json               # simple metadata (rows, source, etc.)

Usage examples:
  # Default out-dir + 100k rows
  python experiments/download_calvin.py

  # Custom out-dir and smaller subset
  python experiments/download_calvin.py --out-dir extdataset/calvin_abc_small --max-samples 20000
"""

import argparse
import json
from pathlib import Path


def download_calvin_abc(out_dir: Path, max_samples: int) -> bool:
    """Download Calvin ABC subset as a single parquet file."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install: pip install pandas pyarrow huggingface_hub")
        return False

    repo_id = "InternRobotics/InternData-Calvin_ABC"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "calvin_abc.parquet"

    if parquet_path.exists():
        print(f"CALVIN ABC parquet already exists at {parquet_path} (delete it to re-download).")
        return True

    print(f"Listing parquet files from {repo_id}...")
    try:
        files = list_repo_files(repo_id, repo_type="dataset", revision="refs/convert/parquet")
        parquet_files = [f for f in files if f.endswith(".parquet")]
    except Exception as e:
        print(f"Failed to list parquet files for {repo_id}: {e}")
        return False

    if not parquet_files:
        print(f"No parquet files found for {repo_id} (refs/convert/parquet).")
        return False

    print(f"Found {len(parquet_files)} parquet file(s). Downloading and concatenating...")
    dfs = []
    for pf in parquet_files:
        try:
            local_p = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
            dfs.append(pd.read_parquet(local_p))
            print(f"  Loaded {pf} ({len(dfs[-1])} rows)")
        except Exception as e:
            print(f"  Failed to load {pf}: {e}")
            continue

    if not dfs:
        print("Failed to load any parquet files for CALVIN ABC.")
        return False

    import pandas as pd  # ensure alias in this scope
    df = pd.concat(dfs, ignore_index=True)
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(df))
        df = df.head(n)
        print(f"Using first {n} rows (max_samples={max_samples}, total={len(dfs[0]) if dfs else 'n/a'})")
    else:
        print(f"Using all {len(df)} rows.")

    print(f"Saving parquet to {parquet_path} ...")
    df.to_parquet(parquet_path)

    meta = {
        "source": repo_id,
        "parquet_files": parquet_files,
        "rows": int(len(df)),
        "max_samples": int(max_samples) if max_samples is not None else None,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {out_dir / 'meta.json'}")
    print("CALVIN ABC download complete.")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Download CALVIN ABC subset from HuggingFace.")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: extdataset/calvin_abc)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum number of rows to keep from CALVIN ABC (default: 100000).",
    )
    args = p.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (proj_root / "extdataset" / "calvin_abc")

    ok = download_calvin_abc(out_dir, max_samples=args.max_samples)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

