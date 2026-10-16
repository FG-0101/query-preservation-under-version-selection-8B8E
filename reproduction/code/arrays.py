#!/usr/bin/env python3
"""Digest the policy recording arrays by dtype, shape and numerical bytes."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--primary", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

archive = np.load(a.primary / "policy_recordings.npz", allow_pickle=False)
digest = {k: {"shape": list(archive[k].shape),
              "dtype": archive[k].dtype.str,
              "sha256": hashlib.sha256(archive[k].tobytes(order="C")).hexdigest()}
          for k in archive.files}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(digest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
