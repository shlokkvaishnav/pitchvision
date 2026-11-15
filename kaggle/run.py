"""Kaggle kernel for spec 03. It unpacks the pitchvision code from the private dataset and runs the training."""

import glob
import os
import subprocess
import sys
import zipfile

commits = glob.glob("/kaggle/input/**/COMMIT", recursive=True)
if not commits:
    raise SystemExit("dataset pitchvision-src is not attached")
base = os.path.dirname(commits[0])
commit = open(commits[0]).read().strip()
# Kaggle unpacks a zip in a dataset into a folder of the same name. Accept both forms.
if os.path.isdir(os.path.join(base, "pitchvision_src")):
    src = os.path.join(base, "pitchvision_src")
else:
    src = "/kaggle/working/code"
    zipfile.ZipFile(os.path.join(base, "pitchvision_src.zip")).extractall(src)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kloppy"], check=True)
env = dict(os.environ, PYTHONPATH=os.path.join(src, "src"), PITCHVISION_COMMIT=commit)
subprocess.run(
    [sys.executable, os.path.join(src, "scripts", "train_masked.py"),
     "--out", "/kaggle/working/results", "--raw", "/kaggle/working/skillcorner", "--epochs", "30"],
    check=True,
    env=env,
)
