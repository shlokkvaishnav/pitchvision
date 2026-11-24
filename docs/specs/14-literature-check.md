# Spec 14 — Literature check and method decision for v2

Status: written on 2026-11-24, before any v2 code or run. It picks the method and the evaluation for spec 15.

## 1. Question
v2 adds a model that samples several plausible paths for each hidden player, and replaces the single holdout with
leave-one-match-out cross-validation over all 10 full-pitch matches. Before building it: what is the strongest published
method for filling in off-screen players, can we run it, and how do papers evaluate it?

## 2. Search
Web search (2026-11-24) for off-screen / missing player imputation in football and other team sports, multi-agent
trajectory imputation and completion, and diffusion models for multi-agent trajectories (2021 to 2026). Papers were read
from their arXiv / journal pages; code was checked on GitHub.

| Work | Method | Hidden whole agents, long gaps? | Uncertainty | Code |
|---|---|---|---|---|
| Graph Imputer (Omidshafiei et al., *Sci. Rep.* 2022; arXiv 2106.04219) | Bidirectional variational LSTM + graph network, 105 matches, simulated camera | Yes, off-screen players | Samples (VRNN) | No |
| UniTraj / Sports-Traj (Xu and Fu, ICLR 2025, arXiv 2405.17680) | Masked VAE with temporal Mamba and agent attention; one model for forecast, imputation, unseen agents | Yes (5 mask types) | VAE samples | Yes (Apache 2.0, needs `mamba-ssm` on CUDA) |
| TranSPORTmer (Capellera et al., ACCV 2024, arXiv 2410.17785) | Set-attention transformer for forecast, imputation, ball inference | Partly | Point | No |
| **U2Diff** (Capellera et al., CVPR 2025, arXiv 2503.18589) and **U2Diffine** (TPAMI 2026, arXiv 2605.10717) | Conditional diffusion: observed states and a mask concatenated with the noisy sample; denoiser = bidirectional temporal Mamba + agent transformer; 50 training steps, 6 DDIM steps; K = 20 samples; noise-NLL term for per-state uncertainty | Yes (UniTraj masks, incl. fully unseen agents) | Samples + per-state variance | No |
| DBHP / MIDAS (Choi et al., ECML-PKDD 2025, arXiv 2408.10878) | Set transformer predicting position, velocity, acceleration, self-ensemble | Random and agent-wise gaps | Point | No |
| Penn, Donnelly, Bhatt (*R. Soc. Open Sci.* 2025, arXiv 2311.14642) | ARMAX on the ball position, interpolation between broadcast frames (Metrica) | Event-level broadcast data | Point | Partial |
| Choi, S. (arXiv 2607.11548, July 2026) | Training-free online methods; best: **role-anchored centroid voting** (B4) on Metrica with a simulated viewport; pitch-control error as the end measure | Yes, off-screen players | None | Yes (MIT) |
| CSDI (Tashiro et al., NeurIPS 2021) | Conditional score-based diffusion for time-series imputation (the general recipe U2Diff follows) | Generic | Samples | Yes |

What the numbers say: on the UniTraj football and soccer benchmarks U2Diff reports a best-of-20 error (minADE20) about
30% lower than UniTraj (Football-U 2.42 vs 3.55 yd; Soccer-U 53.9 vs 94.6 px). It is the strongest published method for
this task family. Its ablation without the uncertainty term (λ = 0) is almost as good on the scene metric (2.37 vs 2.35 yd).
No paper reports calibrated 90% regions per gap length, or proper scoring rules (CRPS / energy score) for sports imputation;
most report best-of-K, which rewards spread and says nothing about calibration.

## 3. Decision
1. **Method: a conditional diffusion model in the U2Diff / CSDI family, on our axial backbone.** This is the plan the owner
   approved, and the literature confirms it is the right family: the best published method is exactly this design
   (observed values + mask + noisy sample in, a temporal block and an agent block per layer, few-step DDIM sampling, K = 20).
   U2Diff has no public code, so we rebuild the recipe rather than run it. Changes, each for a stated reason:
   - Axial attention over time (instead of bidirectional Mamba): it exports to ONNX and runs in onnxruntime-web; Mamba does not.
   - Condition on the Kalman smoother mean and log-std, and diffuse the residual from the Kalman mean: our strongest
     design idea from specs 04 and 05 (the model starts from the classical estimate).
   - Plain noise-prediction loss (U2Diff λ = 0): the uncertainty comes from the samples; the ablation shows the extra term
     adds little to accuracy.
   - Train on the real camera masks of each match (not random masks): the test gaps are long (74% of hidden time in gaps
     of 9.6 s or more), and random masks would rarely produce those.
2. **Published method we also run: role-anchored centroid voting (Choi 2026, B4).** It is the most recent method on the
   same task and data family (Metrica, off-screen players, pitch control), it has public code (MIT), and it needs no
   training, so it runs on every fold at no GPU cost. We reimplement it from `src/impute_bench.py` (EMA role offsets
   against the voted centroid, at least 3 voters, fallbacks to the snapshot formation anchor and to last-seen with decay).
3. **Not run, and why:** Graph Imputer, TranSPORTmer, U2Diff and MIDAS have no public code. UniTraj has code but needs
   `mamba-ssm` (CUDA build) and its own data format, and U2Diff already beats it by about 30% on its own benchmarks; running it
   would cost a large share of the GPU budget to compare against a method that the method we build has already beaten.
   It stays a possible later addition.
4. **Evaluation (spec 15):** leave-one-match-out over the 10 matches, as approved. Point error of the sample mean and
   median; energy score (the 2-D CRPS) for every method that gives a distribution, so the comparison does not reward spread
   the way best-of-K does; 90% coverage per gap-length bucket; a sample-spread check; and pitch-control agreement on
   each held-out match, the end measure of Choi 2026 and of our spec 13.

## 4. Sources
- Omidshafiei et al. 2022, <https://doi.org/10.1038/s41598-022-12547-0>; arXiv <https://arxiv.org/abs/2106.04219>
- Xu and Fu, UniTraj, <https://arxiv.org/abs/2405.17680>; code <https://github.com/colorfulfuture/UniTraj-pytorch>
- Capellera et al., TranSPORTmer, <https://arxiv.org/abs/2410.17785>
- Capellera et al., U2Diff, <https://arxiv.org/abs/2503.18589>; U2Diffine, <https://arxiv.org/abs/2605.10717>
- Choi et al., MIDAS, <https://arxiv.org/abs/2408.10878>
- Penn, Donnelly, Bhatt, <https://arxiv.org/abs/2311.14642>
- Choi 2026, <https://arxiv.org/abs/2607.11548>; code <https://github.com/nowayfootball/offscreen-impute>
- Tashiro et al., CSDI, <https://arxiv.org/abs/2107.03502>
