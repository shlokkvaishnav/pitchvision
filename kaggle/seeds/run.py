"""Kaggle kernel for spec 08 part A. It unpacks the pitchvision code and trains the spec 05 setup with seeds 1 and 2."""

import glob
import os
import subprocess
import sys
import zipfile

commits = glob.glob("/kaggle/input/**/COMMIT", recursive=True)
if not commits:
    raise SystemExit("dataset pitchvision-src-seeds is not attached")
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
for seed in (1, 2):
    subprocess.run(
        [
            sys.executable,
            os.path.join(src, "scripts", "train_more.py"),
            "--out",
            f"/kaggle/working/results/seed{seed}",
            "--raw",
            "/kaggle/working/skillcorner",
            "--epochs",
            "30",
            "--seed",
            str(seed),
        ],
        check=True,
        env=env,
    )
