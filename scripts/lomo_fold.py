"""Spec 15: one leave-one-match-out fold. Trains the transformer and the diffusion model, then scores every method.

Usage: PYTHONPATH=src python scripts/lomo_fold.py --test idsse-J03WQQ --out results/15-lomo [--raw data/raw/skillcorner]
Smoke test (CPU, minutes): add --smoke (few windows, 1 epoch, K = 4). Writes fold-<test>.json, fold-<test>.npz and the
two model weights. Runs on Kaggle GPU from kaggle/lomo/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pitchvision import baselines as bl
from pitchvision import diffusion as df
from pitchvision.anchor import SCALE
from pitchvision.centroid_vote import centroid_vote_by_period
from pitchvision.dataset import concat, load_match
from pitchvision.lomo import GAP_BUCKETS_S, HOLDOUT, all_donors, fold
from pitchvision.model import CHI2_2DOF_90, AnchoredModel, predict_windows
from pitchvision.pitch_control import control_map_fast
from pitchvision.prep import FPS
from pitchvision.train import fit

MID, LAG = 45, 5
BLOCK_FRAMES = int(5 * 60 * FPS)
ETAS = (0.0, 0.5, 1.0)
N_ETA_WINDOWS = 256
POINT = ("spline", "kalman", "centroid_vote", "transformer", "diffusion")


def load(name, donors, raw, max_windows):
    return load_match(name, donors[name], raw, max_windows, allow_holdout=name in HOLDOUT)


def gaussian_es(mean_m, std_m, truth_m, k, rng):
    """Energy score of a diagonal Gaussian from k draws, per cell. Arrays (N, 2) in metres."""
    out = np.empty(len(mean_m))
    for i in range(0, len(mean_m), 100_000):
        s = slice(i, i + 100_000)
        draws = mean_m[s, None] + std_m[s, None] * rng.standard_normal((len(mean_m[s]), k, 2))
        out[s] = df.energy_score(draws, truth_m[s])
    return out


def window_cells(arr, w):
    """Full-match array (T, N, ...) -> window array (B, L, 22, ...)."""
    frames = w.start[:, None] + np.arange(w.pos.shape[1])[None]
    return arr[frames[:, :, None], w.cols[:, None, :]]


def diffusion_scores(model, w, device, eta, k, hidden, truth_m, batch):
    """Sample in chunks; keep per hidden cell scores and the sample mean of every window (normalised units)."""
    parts: dict[str, list] = {key: [] for key in ("err_mean", "err_median", "es", "inside", "spread")}
    mean = np.empty_like(w.pos)
    cell = 0
    for i in range(0, len(w.pos), batch):
        s = slice(i, i + batch)
        smp = df.sample(model, w.pos[s], w.visible[s], w.agent_type, w.imp_mean[s], w.imp_ls[s], k=k, eta=eta,
                        seed=i, batch=batch, device=device)  # fmt: skip
        mean[s] = smp.mean(1)
        h = hidden[s]
        cells = np.moveaxis(smp[:, :, :, :22], 1, 3)[h] * SCALE  # (n, K, 2)
        n = len(cells)
        stats = df.sample_stats(cells, truth_m[cell : cell + n])
        for key, lst in parts.items():
            lst.append(stats[key])
        cell += n
    return {key: np.concatenate(v) for key, v in parts.items()}, mean


def choose_eta(model, tune_w, device, k, batch):
    idx = np.unique(np.linspace(0, len(tune_w.pos) - 1, N_ETA_WINDOWS).round().astype(int))
    sub = type(tune_w)(**{f: getattr(tune_w, f) for f in tune_w.__dataclass_fields__})
    for f in ("pos", "visible", "start", "cols", "imp_mean", "imp_ls", "fc_mean", "fc_ls"):
        setattr(sub, f, getattr(tune_w, f)[idx])
    hidden = ~sub.visible[..., :22]
    truth = (sub.pos[..., :22, :] * SCALE)[hidden]
    scores = {}
    for eta in ETAS:
        st, _ = diffusion_scores(model, sub, device, eta, k, hidden, truth, batch)
        scores[eta] = float(st["es"].mean())
        print(f"eta {eta}: tune energy score {scores[eta]:.3f} m", flush=True)
    return min(scores, key=scores.get), scores


def pitch_control(series, truth_m, ball_m, seen, starts, fps):
    """Agreement with truth at the middle frame of each window with a hidden player (spec 13 setup)."""
    team = np.r_[np.zeros(11, int), np.ones(11, int)]
    rows = {v: [] for v in ["visible", *series]}
    frames = []
    for b in range(len(truth_m)):
        hid = ~seen[b, MID]
        if not hid.any():
            continue

        def cmap(p, idx=slice(None), b=b):
            vel = (p[b, MID] - p[b, MID - LAG]) / (LAG / fps)
            return control_map_fast(p[b, MID][idx], vel[idx], team[idx], ball_m[b, MID]) >= 0.5

        ref = cmap(truth_m)
        rows["visible"].append(float((cmap(truth_m, ~hid) == ref).mean()))
        for v, p in series.items():
            rows[v].append(float((cmap(p) == ref).mean()))
        frames.append(int(starts[b]) + MID)
    return {v: np.asarray(x) for v, x in rows.items()}, np.asarray(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="results/15-lomo")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=df.K_SAMPLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke", action="store_true", help="smoke test only: Metrica, 40 windows per match, 1 epoch, K = 4"
    )
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    max_w, epochs, k = (40, 1, 4) if args.smoke else (0, args.epochs, args.k)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, donors = Path(args.raw), all_donors()
    train_ids, tune_id = fold(args.test)
    if args.smoke:  # smoke test only: Metrica matches, so it runs from the local cache
        others = [m for m in ("metrica-1", "metrica-2", "metrica-3") if m != args.test]
        train_ids, tune_id = others[:1], others[1]
    print(f"fold test {args.test}, tune {tune_id}, train {train_ids}", flush=True)
    t0 = time.time()

    train_w = concat([load(m, donors, raw, max_w)[2] for m in train_ids])
    gc.collect()
    tune = load(tune_id, donors, raw, max_w)
    tune_w = tune[2]
    windows = {"train": len(train_w.pos), "tune": len(tune_w.pos)}

    # transformer (spec 05 recipe, 20 epochs, early stopping and std scale on the tuning match)
    tf = AnchoredModel(length=90, dim=128, heads=4, depth=4).to(device)
    tf_log, tf_best = fit(tf, train_w, device, epochs, args.batch, args.lr, args.seed, tune=[tune], amp=True)
    hid_t = ~tune_w.visible[..., :22]
    mu, ls = predict_windows(tf, tune_w.pos, tune_w.visible, tune_w.agent_type, tune_w.imp_mean, tune_w.imp_ls, device)
    z2 = (((mu[..., :22, :] - tune_w.pos[..., :22, :]) / np.exp(ls[..., :22, :])) ** 2)[hid_t]
    log_scale = float(0.5 * np.log(np.mean(z2)))
    t_tf = time.time() - t0

    # diffusion
    torch.manual_seed(args.seed)
    dm = df.Denoiser(length=90, dim=128, heads=4, depth=4).to(device)
    dm_log, dm_best = df.fit(dm, train_w, device, epochs, args.batch, args.lr, args.seed, tune=tune_w)
    del train_w
    gc.collect()
    sample_batch = 16 if device == "cuda" else 4
    eta, eta_scores = choose_eta(dm, tune_w, device, k, sample_batch)
    del tune, tune_w
    gc.collect()
    t_train = time.time() - t0

    # the test match: loaded only now, after both models are final
    tr, vis, w = load(args.test, donors, raw, max_w)
    hidden = ~w.visible[..., :22]
    truth_w = w.pos[..., :22, :] * SCALE
    truth = truth_w[hidden]
    est = {
        "spline": bl.by_period(bl.cubic_spline, tr.players, vis, tr.period),
        "centroid_vote": centroid_vote_by_period(tr.players, vis, tr.teams, tr.period),
    }
    kal_mean = np.full(tr.players.shape, np.nan)
    kal_std = np.full(tr.players.shape, np.nan)
    gap = np.zeros(vis.shape, int)
    for p in np.unique(tr.period):
        rows = tr.period == p
        kk = bl.kalman(tr.players[rows], vis[rows], FPS)
        kal_mean[rows] = kk.smooth_x[..., 0]
        kal_std[rows] = np.sqrt(np.maximum(kk.smooth_p[..., 0, 0], 1e-6))
        gap[rows] = bl.gap_lengths(~vis[rows])
    est["kalman"] = kal_mean
    est_w = {m: window_cells(e, w) for m, e in est.items()}

    mu, ls = predict_windows(tf, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
    mu_m = mu[..., :22, :] * SCALE
    sd_m = np.exp(ls[..., :22, :] + log_scale) * SCALE
    dstats, dmean = diffusion_scores(dm, w, device, eta, k, hidden, truth, sample_batch)
    dmean_m = dmean[..., :22, :] * SCALE

    rng = np.random.default_rng(0)
    cells = {
        "err_spline": np.linalg.norm(est_w["spline"][hidden] - truth, axis=-1),
        "err_kalman": np.linalg.norm(est_w["kalman"][hidden] - truth, axis=-1),
        "err_centroid_vote": np.linalg.norm(est_w["centroid_vote"][hidden] - truth, axis=-1),
        "err_transformer": np.linalg.norm(mu_m[hidden] - truth, axis=-1),
        "err_diffusion": dstats["err_mean"],
        "err_diffusion_median": dstats["err_median"],
        "es_kalman": gaussian_es(est_w["kalman"][hidden], window_cells(kal_std, w)[hidden], truth, k, rng),
        "es_transformer": gaussian_es(mu_m[hidden], sd_m[hidden], truth, k, rng),
        "es_diffusion": dstats["es"],
        "inside_kalman": ((((est_w["kalman"] - truth_w) / window_cells(kal_std, w)) ** 2).sum(-1) <= CHI2_2DOF_90)[
            hidden
        ],
        "inside_transformer": ((((mu_m - truth_w) / sd_m) ** 2).sum(-1) <= CHI2_2DOF_90)[hidden],
        "inside_diffusion": dstats["inside"],
        "spread_diffusion": dstats["spread"],
        "gap_s": window_cells(gap, w)[hidden] / FPS,
        "block": np.broadcast_to((w.start // BLOCK_FRAMES)[:, None, None], hidden.shape)[hidden],
    }
    ok = np.all([np.isfinite(cells[f"err_{m}"]) for m in POINT], axis=0) & np.isfinite(cells["es_kalman"])
    cells = {key: v[ok] for key, v in cells.items()}

    seen = w.visible[..., :22]
    series = {
        "spline": np.where(seen[..., None], truth_w, est_w["spline"]),
        "kalman": np.where(seen[..., None], truth_w, est_w["kalman"]),
        "centroid_vote": np.where(seen[..., None], truth_w, est_w["centroid_vote"]),
        "transformer": np.where(seen[..., None], truth_w, mu_m),
        "diffusion": np.where(seen[..., None], truth_w, dmean_m),
    }
    pc, pc_frames = pitch_control(series, truth_w, w.pos[..., 22, :] * SCALE, seen, w.start, FPS)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fold-{args.test}" + ("-smoke" if args.smoke else "")
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        **{key: v.astype(np.float32) if v.dtype.kind == "f" else v for key, v in cells.items()},
        **{f"pc_{v}": x.astype(np.float32) for v, x in pc.items()},
        pc_block=pc_frames // BLOCK_FRAMES,
    )
    gaps = np.digitize(cells["gap_s"], GAP_BUCKETS_S[1:-1])
    summary = {
        "spec": "15-leave-one-match-out",
        "commit": args.commit,
        "device": device,
        "smoke": args.smoke,
        "test": args.test,
        "tune": tune_id,
        "train": train_ids,
        "settings": {**vars(args), "epochs_used": epochs, "k_used": k},
        "windows": {**windows, "test": len(w.pos)},
        "train_seconds": round(t_train, 1),
        "transformer_seconds": round(t_tf, 1),
        "total_seconds": round(time.time() - t0, 1),
        "transformer": {"best_epoch": tf_best, "std_scale": float(np.exp(log_scale)), "log": tf_log},
        "diffusion": {"best_epoch": dm_best, "eta": eta, "eta_tune_energy_score_m": eta_scores, "log": dm_log},
        "n_scored_cells": int(ok.sum()),
        "n_hidden_cells": int(ok.size),
        "median_m": {m: float(np.median(cells[f"err_{m}"])) for m in POINT},
        "mean_m": {m: float(np.mean(cells[f"err_{m}"])) for m in POINT},
        "diffusion_sample_median_median_m": float(np.median(cells["err_diffusion_median"])),
        "energy_score_m": {m: float(np.mean(cells[f"es_{m}"])) for m in ("kalman", "transformer", "diffusion")},
        "coverage_90": {m: float(np.mean(cells[f"inside_{m}"])) for m in ("kalman", "transformer", "diffusion")},
        "coverage_90_by_gap": {
            m: [float(np.mean(cells[f"inside_{m}"][gaps == g])) if (gaps == g).any() else None for g in range(5)]
            for m in ("kalman", "transformer", "diffusion")
        },
        "pitch_control_frames": len(pc_frames),
        "pitch_control_agreement": {v: float(x.mean()) if len(x) else None for v, x in pc.items()},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    torch.save(tf.state_dict(), out_dir / f"{stem}-transformer.pt")
    torch.save(dm.state_dict(), out_dir / f"{stem}-diffusion.pt")
    print(json.dumps({key: summary[key] for key in ("median_m", "coverage_90", "pitch_control_agreement")}, indent=2))


if __name__ == "__main__":
    main()
