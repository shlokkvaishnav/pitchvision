"""Kaggle kernel for spec 15, one leave-one-match-out fold. It unpacks the pitchvision code and runs scripts/lomo_fold.py."""

import glob
import os
import subprocess
import sys
import zipfile

TEST = "idsse-J03WQQ"
commits = glob.glob("/kaggle/input/**/COMMIT", recursive=True)
if not commits:
    raise SystemExit("dataset pitchvision-src-lomo is not attached")
base = os.path.dirname(commits[0])
with open(commits[0]) as fh:
    commit = fh.read().strip()
# Kaggle unpacks a zip in a dataset into a folder of the same name. Accept both forms.
if os.path.isdir(os.path.join(base, "pitchvision_src")):
    src = os.path.join(base, "pitchvision_src")
else:
    src = "/kaggle/working/code"
    zipfile.ZipFile(os.path.join(base, "pitchvision_src.zip")).extractall(src)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kloppy"], check=True)
env = dict(os.environ, PYTHONPATH=os.path.join(src, "src"), PITCHVISION_COMMIT=commit)
script = os.path.join(src, "scripts", "lomo_fold.py")
subprocess.run(
    [sys.executable, script, "--test", TEST, "--out", "/kaggle/working/results", "--raw", "/kaggle/working/skillcorner"],
    check=True,
    env=env,
)
