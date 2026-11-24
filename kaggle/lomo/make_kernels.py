"""Write one Kaggle kernel folder per spec 15 fold (kaggle/lomo/<match>/). Run from the repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from pitchvision.lomo import MATCHES  # noqa: E402

here = Path(__file__).resolve().parent
template = (here / "run_template.py").read_text()
for m in MATCHES:
    d = here / m
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text(template.replace("__TEST__", m))
    slug = f"pitchvision-lomo-{m.lower()}"
    meta = {
        "id": f"shlokkvaishnav/{slug}", "title": slug.replace("-", " "), "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": ["shlokkvaishnav/pitchvision-src-lomo"],
        "competition_sources": [], "kernel_sources": [],
    }  # fmt: skip
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
