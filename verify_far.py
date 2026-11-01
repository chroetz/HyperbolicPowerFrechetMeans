"""Independent check of the far-out minimizers: recompute every replication with d(o, m_n) > 25 for the given
n and alphas with the high-precision Newton method (mp_newton.py, adaptive precision; started at the stored
radius in the direction of the farthest observation) and report the maximal relative difference.
Usage: python3 verify_far.py N1,N2,... ALPHA1,ALPHA2,... results1.npz [results2.npz ...]
(31 Oct 2026: n in {4,...,1024}, alpha in {1.6, 2} (and 1.1, 1.9 for n = 4, 32): 1981 runs, max rel. diff 1.2e-7,
i.e. float32 storage precision; for n = 2, the exact midpoint formula agrees for all 10^5 replications.)"""
import numpy as np, sys, time
from h2sim import sample
from mp_newton import frechet_mean_mp
ns = [int(x) for x in sys.argv[1].split(',')]; al = [float(a) for a in sys.argv[2].split(',')]
z = {}
for f in sys.argv[3:]: z.update(dict(np.load(f)))
for n in ns:
    rng = np.random.default_rng([20261031, n]); R = len(z[f'err_{al[0]:g}_{n}'])
    chunk = 2000 if n <= 64 else (1000 if n <= 256 else 500)
    worst, cnt, t0 = 0.0, 0, time.time()
    for c in range(0, R, chunk):
        ry, ty = sample(rng, min(chunk, R - c), n)
        for a in al:
            st = z[f'err_{a:g}_{n}'][c:c + chunk].astype(float)
            for i in np.where(st > 25)[0]:
                j = int(np.argmax(ry[i]))
                val = frechet_mean_mp(ry[i], ty[i], a, start=(st[i], ty[i, j]))[0]
                rel = abs(val - st[i]) / val; worst = max(worst, rel); cnt += 1
                if rel > 1e-6: print('MISMATCH', a, n, c + i, st[i], val, flush=True)
    print(f'n={n} alphas={al}: checked {cnt} runs with d(o,m_n)>25, max rel diff {worst:.2e}, {time.time()-t0:.0f}s', flush=True)
