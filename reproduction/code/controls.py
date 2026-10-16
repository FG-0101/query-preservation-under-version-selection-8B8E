#!/usr/bin/env python3
"""Current-extractor control: transformed-file minus source-WAV descriptors."""
from pathlib import Path
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

rows = json.loads(a.data.read_text())["rows"]
controls = []
for signal in ["two_tone", "amplitude_modulated", "chirp"]:
    group = {r["transform"]: r for r in rows if r["signal"] == signal}
    for transform in ["aiff", "aac"]:
        controls.append({
            "signal": signal,
            "transform": transform,
            "reported_encoded_md5_changed":
                group[transform]["reported_md5_encoded"] != group["wav"]["reported_md5_encoded"],
            "selected_features_exact_equal":
                group[transform]["features"] == group["wav"]["features"],
            "feature_differences":
                {f: group[transform]["features"][f] - group["wav"]["features"][f]
                 for f in group["wav"]["features"]},
        })
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(controls, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
