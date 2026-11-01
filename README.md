# HyperbolicPowerFrechetMeans

Simulation code for Section 6 ("Numerical Illustration in the Hyperbolic Plane") of

> Christof Schötz. *Transformed Fréchet Means for Robust Estimation in Hadamard Spaces.*
> [arXiv:2511.06933](https://arxiv.org/abs/2511.06933)

The code estimates the risk of sample power Fréchet means (α-Fréchet means) in the hyperbolic plane for
heavy-tailed data with infinite variance, and produces Figure 3 of the paper.

## Setting

- Space: the hyperbolic plane H² (constant curvature −1) with a fixed point o.
- Data: Y = exp_o(V), where V follows an isotropic bivariate Student-t distribution with ν = 3/2 degrees of
  freedom on the tangent space T_o H² ≅ ℝ². Then d(o, Y) = ‖V‖, so E[d(o, Y)^p] < ∞ if and only if p < 3/2.
  In particular, Y has infinite variance.
- Target: by rotational symmetry and uniqueness, the α-Fréchet mean of Y is m = o for every α ∈ (1, 2]. Hence
  the error d(m, m_n) = d(o, m_n) of the sample α-Fréchet mean m_n is known exactly.
- Design: α ∈ {1.1, 1.2, …, 2.0}, n ∈ {1, 2, 4, …, 1024}, and 10⁵ independent samples for each n. For each
  sample, m_n is computed for every α from the same data (common random numbers).

## Computation

- Points are stored in geodesic polar coordinates (r, θ) about o. Distances, gradients, and the exponential map
  use overflow-free formulas, so arbitrarily distant observations are handled.
- The sample α-Fréchet mean m_n = argmin_q Σᵢ d(Yᵢ, q)^α is computed by a damped Riemannian Newton method with
  Armijo backtracking along geodesics, started at the coordinatewise median of log_o(Y₁), …, log_o(Y_n).
  The objective is geodesically convex, and strictly so for α > 1. For n = 1, m₁ = Y₁.
- The iteration stops when the Newton decrement is small, after at most 1000 iterations. If it has not
  converged by then, the a-posteriori bound |grad F| / λ_min(Hess F) on the distance to the minimizer is
  evaluated. If this bound exceeds 10⁻⁸ · d(o, m_n), m_n is recomputed with a high-precision Newton method
  (`mp_newton.py`, based on mpmath, with the working precision adapted to d(o, m_n)), started at the last
  iterate.
- In the run reported in the paper, 205 of the 10⁷ Newton runs did not converge within 1000 iterations,
  175 of them for α = 1.1. For 174 of the 205, the a-posteriori bound was already below the threshold; the
  other 31 were recomputed in high precision.
- As an independent check, `verify_far.py` recomputes all replications with d(o, m_n) > 25 in high precision.
  For the run in the paper (α ∈ {1.6, 2} for n ∈ {4, …, 1024}, and also α ∈ {1.1, 1.9} for n ∈ {4, 32}), this
  covered 1981 runs; the maximal relative difference was 1.2·10⁻⁷, which is the precision of the float32
  storage.

## Files

| File | Content |
| --- | --- |
| `h2sim.py` | Simulation: sampling, batched Newton method, high-precision fallback. Writes an `.npz` file. |
| `mp_newton.py` | High-precision (mpmath) damped Riemannian Newton method, used by `h2sim.py` and `verify_far.py`. |
| `plot.py` | Figure 3 and summary tables from one or more `.npz` result files. |
| `verify_far.py` | Independent high-precision check of the far-out minimizers. |
| `h2_part1.npz`, `h2_part2.npz` | Results of the run in the paper: arrays `err_<alpha>_<n>` of the errors d(o, m_n) (float32, one per replication) and a JSON string `info` with iteration and convergence counts. |
| `fig_h2.pdf`, `fig_h2.png` | Figure 3 of the paper: L¹-risk and Lᵅ-risk (for α < 3/2) against n. |
| `fig_h2_mc.pdf`, `fig_h2_mc.png` | Diagnostic: running Monte Carlo estimate of the Lᵅ-risk at n = 1024. For α ≥ 3/2, the Lᵅ-risk is infinite, so this estimate is not meaningful. |
| `fig_h2_summary.csv` | Least-squares slopes of the log-risks against log n for n ≥ 64, and iteration counts. |
| `fig_h2_risks.csv` | Risk estimates (L¹ with standard error, Lᵅ, median, 90% quantile, maximum) for each α and n. |

## Requirements

Python 3 with NumPy, Matplotlib, and mpmath (`pip install -r requirements.txt`). The results were produced
with Python 3.13, NumPy 2.5, Matplotlib 3.11, and mpmath 1.4.

## Reproducing the results

Run the simulation (about 35 minutes with two processes on two cores):

```sh
A=1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0
python3 h2sim.py $A 1024,32,8,2,1 100000 h2_part1.npz &
python3 h2sim.py $A 512,256,128,64,16,4 100000 h2_part2.npz &
wait
```

The arguments of `h2sim.py` are the values of α, the sample sizes n, the number of replications, the output
file, and optionally a seed (default 20261031). Each n uses its own random stream
`numpy.random.default_rng([seed, n])`, so the results do not depend on how the sample sizes are split across
processes. In the environment above, rerunning reproduces the stored arrays bit for bit (checked for n = 2
and n = 8).

Create the figure and the tables from the stored or recomputed results (a few seconds):

```sh
python3 plot.py fig_h2 h2_part1.npz h2_part2.npz
```

Optional arguments: `--fitmin N` sets the smallest n used for the fitted slopes (default 64), and
`--cmap NAME LIGHT DARK` sets the colormap (default `plasma 0.8 0.0`).

Run the independent check of the far-out minimizers:

```sh
python3 verify_far.py 4,8,16,32,64,128,256,512,1024 1.6,2 h2_part1.npz h2_part2.npz
python3 verify_far.py 4,32 1.1,1.9 h2_part1.npz h2_part2.npz
```

## Citation

```bibtex
@misc{Schoetz2025TransformedFrechet,
  author        = {Sch{\"o}tz, Christof},
  title         = {Transformed {F}r{\'e}chet Means for Robust Estimation in {H}adamard Spaces},
  year          = {2025},
  eprint        = {2511.06933},
  archivePrefix = {arXiv}
}
```
