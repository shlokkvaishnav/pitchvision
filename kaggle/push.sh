#!/bin/sh
# Pack the code at HEAD into a private Kaggle dataset, then push a kernel that uses it.
# Usage: [KDIR=kaggle/audit] [DS=pitchvision-src-cpu] sh kaggle/push.sh "version message"
# Each kernel can have its own dataset (DS), so that a queued kernel does not get a newer code version.
set -eu
K="${KAGGLE:-$HOME/.cache/kaggle-cli/bin/kaggle}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="${DS:-pitchvision-src}"
KDIR="${KDIR:-$ROOT/kaggle}"
OUT="${TMPDIR:-/tmp}/$DS"
grep -q "shlokkvaishnav/$DS\"" "$KDIR/kernel-metadata.json" || { echo "kernel-metadata.json does not use dataset $DS"; exit 1; }
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$ROOT"
git diff --quiet HEAD || { echo "commit your changes first"; exit 1; }
git archive --format=zip -o "$OUT/pitchvision_src.zip" HEAD src scripts
git rev-parse HEAD > "$OUT/COMMIT"
cat > "$OUT/dataset-metadata.json" <<META
{"title": "$DS", "id": "shlokkvaishnav/$DS", "licenses": [{"name": "other"}]}
META
if "$K" datasets status "shlokkvaishnav/$DS" >/dev/null 2>&1; then
  "$K" datasets version -p "$OUT" -m "${1:-update}"
else
  "$K" datasets create -p "$OUT"
fi
for i in $(seq 1 30); do
  "$K" datasets status "shlokkvaishnav/$DS" 2>/dev/null | grep -q ready && break
  sleep 10
done
sleep 20
"$K" kernels push -p "$KDIR"
