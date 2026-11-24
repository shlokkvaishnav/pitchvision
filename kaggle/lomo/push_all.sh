#!/bin/sh
# Spec 15: pack the code at HEAD into the private dataset pitchvision-src-lomo once, then push every fold kernel.
# Usage: sh kaggle/lomo/push_all.sh ["version message"] [match ...]   (no matches = all 10)
# All folds must run from one commit: do not re-version the dataset while a fold kernel is queued or running.
set -eu
K="${KAGGLE:-$HOME/.cache/kaggle-cli/bin/kaggle}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DS=pitchvision-src-lomo
OUT="${TMPDIR:-/tmp}/$DS"
cd "$ROOT"
git diff --quiet HEAD || { echo "commit your changes first"; exit 1; }
MSG="${1:-spec 15}"
[ $# -gt 0 ] && shift
if [ "${SKIP_DATASET:-0}" != 1 ]; then
  rm -rf "$OUT" && mkdir -p "$OUT"
  git archive --format=zip -o "$OUT/pitchvision_src.zip" HEAD src scripts
  git rev-parse HEAD > "$OUT/COMMIT"
  printf '{"title": "%s", "id": "shlokkvaishnav/%s", "licenses": [{"name": "other"}]}\n' "$DS" "$DS" > "$OUT/dataset-metadata.json"
  if "$K" datasets status "shlokkvaishnav/$DS" >/dev/null 2>&1; then
    "$K" datasets version -p "$OUT" -m "$MSG"
  else
    "$K" datasets create -p "$OUT"
  fi
  for i in $(seq 1 30); do
    "$K" datasets status "shlokkvaishnav/$DS" 2>/dev/null | grep -q ready && break
    sleep 10
  done
  sleep 20
fi
if [ $# -eq 0 ]; then set -- $(ls -d kaggle/lomo/*/ | xargs -n1 basename); fi
for m in "$@"; do
  "$K" kernels push -p "kaggle/lomo/$m" || echo "push failed: $m"
done
