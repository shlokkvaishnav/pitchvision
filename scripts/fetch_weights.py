"""Download the trained spec 05 weights from the GitHub release and check their SHA-256.

Usage: python scripts/fetch_weights.py
Writes data/models/05-more-data.pt. Training them again needs a GPU (scripts/train_more.py, about 1 hour on Kaggle).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/shlokkvaishnav/pitchvision/releases/download/v1.0/05-more-data.pt"
SHA256 = "59325ef022e2cfab35e1d0bcbd8a61ecf6c1e1eb662732724f2354f220ce7560"
OUT = ROOT / "data" / "models" / "05-more-data.pt"


def main() -> None:
    if not OUT.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            OUT.write_bytes(r.read())
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{OUT} has SHA-256 {digest}, expected {SHA256}")
    print(f"{OUT.relative_to(ROOT)}: SHA-256 ok")


if __name__ == "__main__":
    main()
