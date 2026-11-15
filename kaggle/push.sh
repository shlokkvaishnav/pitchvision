#!/bin/sh
# Pack the code at HEAD into the private dataset shlokkvaishnav/pitchvision-src, then push the kernel.
# Usage: sh kaggle/push.sh "version message"
set -eu
K="${KAGGLE:-$HOME/.cache/kaggle-cli/bin/kaggle}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/pitchvision-src"
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$ROOT"
git diff --quiet HEAD || { echo "commit your changes first"; exit 1; }
git archive --format=zip -o "$OUT/pitchvision_src.zip" HEAD src scripts
git rev-parse HEAD > "$OUT/COMMIT"
cat > "$OUT/dataset-metadata.json" <<META
{"title": "pitchvision-src", "id": "shlokkvaishnav/pitchvision-src", "licenses": [{"name": "other"}]}
META
if "$K" datasets status shlokkvaishnav/pitchvision-src >/dev/null 2>&1; then
  "$K" datasets version -p "$OUT" -m "${1:-update}"
else
  "$K" datasets create -p "$OUT"
fi
sleep 30
"$K" kernels push -p "$ROOT/kaggle"
