# Spec 15, fold 1 of 5: held-out match idsse-J03WMX

Status: **complete (one fold only)**. Folds 2 to 5 have not been run and are deferred. This is one match, so it is not
the 5-fold headline and should not be quoted as one.

- Kernel: `shlokkvaishnav/pitchvision-lomo-idsse-j03wmx`, status COMPLETE, run on Kaggle GPU (cuda), 6,217 s total.
- Code commit: `8f82af3` (spec 15 amendment A2). Settings: 20 epochs, batch 32, lr 3e-4, K = 20 diffusion samples, seed 0.
- Training matches: 8 (`train`), tuning match `idsse-J03WOH`, test match `idsse-J03WMX`. Windows: 14,272 train, 2,703 tune, 2,883 test.
- Both learned models reached their best tuning score at the last epoch (epoch 20). They may be undertrained.

## Result (test match only, metres unless noted)

| Method | Median error | Mean error | Energy score | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | 3.03 | 8.48 | n/a | n/a | 0.920 |
| Kalman | 3.55 | 10.00 | 11.16 | 0.892 | 0.908 |
| Centroid vote | 8.89 | 12.48 | n/a | n/a | 0.845 |
| Transformer (v1) | **2.60** | **4.90** | **3.57** | 0.951 | **0.936** |
| Diffusion (v2) | 2.89 | 5.35 | 3.91 | 0.809 | 0.931 |

Visible-only pitch-control agreement is 0.712 (the baseline that the hidden players are added to).

Reading: on this match the transformer is better than the diffusion model on every error and agreement measure. The
diffusion model's 90% regions are too narrow (coverage 0.81 against a target of 0.90). The diffusion model did not beat
the transformer here, so the v2 claim is not supported by this fold.

## Files

- `fold-idsse-J03WMX.json`: full summary (training logs, metrics, per-gap coverage, timings).
- Not in git, kept in the private archive (`private-archives/pitchvision-lomo/idsse-J03WMX/`): the prediction arrays
  `fold-idsse-J03WMX.npz` (84 MB), the two model checkpoints (`*-transformer.pt`, `*-diffusion.pt`), and the SkillCorner
  inputs it read (raw tracking files; reproducible from the same public source).
- Kernel script and metadata: `kaggle/lomo/idsse-J03WMX/`.

## Resume

Nothing from this fold is on the resume. The approved bullets are unchanged.
