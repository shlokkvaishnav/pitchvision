"""Kaggle kernel for spec 03. It unpacks the pitchvision code from the private dataset and runs the training."""

import glob
import os
import subprocess
import sys
import zipfile

zips = glob.glob("/kaggle/input/**/pitchvision_src.zip", recursive=True)
if not zips:
    raise SystemExit("dataset pitchvision-src is not attached")
src = "/kaggle/working/code"
zipfile.ZipFile(zips[0]).extractall(src)
commit = open(os.path.join(os.path.dirname(zips[0]), "COMMIT")).read().strip()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kloppy"], check=True)
env = dict(os.environ, PYTHONPATH=os.path.join(src, "src"), PITCHVISION_COMMIT=commit)
subprocess.run(
    [sys.executable, os.path.join(src, "scripts", "train_masked.py"),
     "--out", "/kaggle/working/results", "--raw", "/kaggle/working/skillcorner", "--epochs", "30"],
    check=True,
    env=env,
)
