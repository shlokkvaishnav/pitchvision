"""Spec 09 note (exploratory): smaller ONNX files for the page. fp16 and int8 against the fp32 site model.

Two steps, because the conversion tools are not project dependencies:
  1. PYTHONPATH=src .venv/bin/python scripts/onnx_variants.py dump
     Writes the windows of the spec 05 test matches (inputs and truth) to runs/onnx-variants/windows.npz.
  2. python scripts/onnx_variants.py eval   (an environment with onnx, onnxruntime, onnxconverter-common)
     Writes runs/onnx-variants/model-{fp16,int8}.onnx, scores each variant on the dumped windows (onnxruntime, CPU),
     times onnxruntime-web (wasm) in Node on the demo clips, and writes results/09-onnx-variants.json.
The site keeps the fp32 model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "onnx-variants"
TEST = {"J03WQQ": "1959846", "J03WR9": "1986691"}  # spec 05 test matches and donors (not the holdout)
STD_SCALE = 1.1747
CHI2_90 = 4.605170185988091
SCALE = np.array([52.5, 34.0])


def dump() -> None:
    from pitchvision.dataset import load_match

    RUNS.mkdir(parents=True, exist_ok=True)
    parts = [load_match(f"idsse-{m}", d, ROOT / "data" / "raw" / "skillcorner")[2] for m, d in TEST.items()]
    np.savez_compressed(
        RUNS / "windows.npz",
        pos=np.concatenate([w.pos for w in parts]).astype(np.float32),
        visible=np.concatenate([w.visible for w in parts]),
        agent_type=parts[0].agent_type,
        imp_mean=np.concatenate([w.imp_mean for w in parts]).astype(np.float32),
        imp_ls=np.concatenate([w.imp_ls for w in parts]).astype(np.float32),
    )


def run_all(sess, d, batch: int = 1):  # the site model has a fixed batch of 1
    mus, lss = [], []
    for i in range(0, len(d["pos"]), batch):
        s = slice(i, i + batch)
        obs = d["visible"][s]
        mu, ls = sess.run(None, {
            "pos": np.nan_to_num(d["pos"][s]), "observed": obs.astype(np.float32),
            "agent_type": d["agent_type"].astype(np.int64), "anchor_mean": d["imp_mean"][s],
            "anchor_log_std": d["imp_ls"][s],
        })  # fmt: skip
        mus.append(mu)
        lss.append(ls)
    return np.concatenate(mus), np.concatenate(lss)


def evaluate() -> None:
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32 = ROOT / "site" / "demo" / "model.onnx"
    paths = {"fp32": fp32, "fp16": RUNS / "model-fp16.onnx", "int8": RUNS / "model-int8.onnx"}
    m = onnx.load(fp32)
    casts = [n.name for n in m.graph.node if n.op_type == "Cast"]  # converted Cast nodes break the type check
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True, node_block_list=casts), paths["fp16"])
    quantize_dynamic(fp32, paths["int8"], weight_type=QuantType.QInt8)

    d = dict(np.load(RUNS / "windows.npz"))
    hidden = ~d["visible"][..., :22]
    truth = d["pos"][..., :22, :]
    out: dict = {"spec": "09 note: ONNX variants (exploratory)", "windows": len(d["pos"]),
                 "hidden_cells": int(hidden.sum()), "variants": {}}  # fmt: skip
    ref = None
    for name, p in paths.items():
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        mu, ls = run_all(sess, d)
        one = {k: v[:1] for k, v in d.items() if k != "agent_type"} | {"agent_type": d["agent_type"]}
        run_all(sess, one)
        t0 = time.perf_counter()
        for _ in range(50):
            run_all(sess, one)
        err = np.linalg.norm((mu[..., :22, :] - truth) * SCALE, axis=-1)[hidden]
        z2 = (((mu[..., :22, :] - truth) / (np.exp(ls[..., :22, :]) * STD_SCALE)) ** 2).sum(-1)[hidden]
        row = {
            "bytes": p.stat().st_size,
            "cpu_ms_per_window": (time.perf_counter() - t0) / 50 * 1000,
            "median_error_m": float(np.median(err)), "mean_error_m": float(err.mean()),
            "coverage_90": float((z2 <= CHI2_90).mean()),
        }  # fmt: skip
        if ref is None:
            ref = mu
        else:
            row["max_abs_mean_diff_vs_fp32_m"] = float((np.abs(mu - ref) * SCALE).max())
        out["variants"][name] = row
    bench = subprocess.run(["node", str(ROOT / "scripts" / "bench_web.mjs"), *map(str, paths.values())],
                           cwd=ROOT, capture_output=True, text=True, check=True)  # fmt: skip
    for name, ms in zip(paths, json.loads(bench.stdout)):
        out["variants"][name]["web_wasm_ms_per_clip"] = ms
    out["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    (ROOT / "results" / "09-onnx-variants.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"dump": dump, "eval": evaluate}[sys.argv[1]]()
