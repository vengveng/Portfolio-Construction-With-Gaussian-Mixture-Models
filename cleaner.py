#!/usr/bin/env python3
import sys
from pathlib import Path

# 1. Point this at the top‐level directory containing all subfolders of
#    interest.
ROOT = Path("data/portfolio_returns")

if not ROOT.exists() or not ROOT.is_dir():
    print(f"Error: {ROOT} does not exist or is not a directory.", file=sys.stderr)
    sys.exit(1)

# 2. Recursively find all .csv files under ROOT.
for csv_path in ROOT.rglob("*.csv"):
    name = csv_path.name

    # 3. If the filename does NOT contain "_Nmax_", delete it.
    if "_Nmax_" not in name:
        try:
            csv_path.unlink()
            print(f"Deleted: {csv_path}")
        except Exception as e:
            print(f"Could not delete {csv_path}: {e}", file=sys.stderr)