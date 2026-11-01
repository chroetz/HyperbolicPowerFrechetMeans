"""
Power Frechet means in the hyperbolic plane H^2 (curvature -1).

Data: Y = exp_o(V), V isotropic bivariate Student-t with nu = 1.5 degrees of
freedom, so d(o, Y) = |V| and E[d(o,Y)^p] < infinity iff p < 1.5.  The law is
invariant under all rotations about o, hence every tau-Frechet mean equals o.

Points are stored in geodesic polar coordinates (r, theta) about o; distances
and gradients use overflow-free formulas, so arbitrarily distant observations
(heavy tails) are handled exactly.  The sample alpha-Frechet mean
    m_n = argmin_q  sum_i d(Y_i, q)^alpha
is computed by a damped Riemannian Newton method (the objective is geodesically
convex, strictly so for alpha > 1); for n = 1, m_1 = Y_1.  For each replication, one sample is drawn
and m_n is computed for every alpha (common random numbers).

In double precision, a point at distance r from o is resolved transversally only to about sinh(r) * 1e-16.  The
iteration occasionally stalls (about 2 in 10^5 runs), when the minimizer is extremely close to an observation
(alpha near 1, harmless) or far from o.  For each
replication that has not converged after 1000 iterations, the a-posteriori error bound |grad F| / lambda_min(Hess F)
is evaluated; if it exceeds 1e-8 * d(o, m_n), m_n is recomputed with the high-precision Newton method of
mp_newton.py (requires mpmath), started at the last iterate.

Usage (settings of Section 6, run on 31 Oct 2026; about 35 min with two processes on 2 cores):
  A=1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0
  python3 h2sim.py $A 1024,32,8,2,1 100000 h2_part1.npz &
  python3 h2sim.py $A 512,256,128,64,16,4 100000 h2_part2.npz &
  python3 plot.py fig_h2 h2_part1.npz h2_part2.npz
Output: an .npz file with arrays err_<alpha>_<n> (float32, the distances d(o, m_n), one per
replication) and a JSON string 'info' with, per (alpha, n), the maximal number of float64 Newton iterations,
the number of non-converged float64 runs, and the number of those recomputed in high precision.
"""
import numpy as np, time, sys, json

LOG2 = np.log(2.0)

def logsinh(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        big = x > 20
        out = np.where(big, x - LOG2 + np.log1p(-np.exp(-2 * np.where(big, x, 20.0))),
                       np.log(np.sinh(np.where(big, 1.0, x))))
    return out

def geom(rq, tq, ry, ty):
    """distance d(q, y) and unit gradient of d(., y) at q in the frame (e_r, e_t)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        Dl = ty - tq
        s2 = np.sin(0.5 * Dl) ** 2
        a = 2 * logsinh(0.5 * np.abs(ry - rq))
        b = logsinh(ry) + logsinh(rq) + np.log(s2)
        L = np.logaddexp(a, b)                         # log sinh^2(d/2)
        d = np.where(L > 40, L + 2 * LOG2, 2 * np.arcsinh(np.exp(np.minimum(L, 40) / 2)))
        lsd = logsinh(d)
        Rs = np.exp(logsinh(ry) - lsd)
        Rc = np.sign(ry - rq) * np.exp(logsinh(np.abs(ry - rq)) - lsd)
        logcosh_rq = rq - LOG2 + np.log1p(np.exp(-2 * rq))
        gr = -Rc + 2 * np.exp(logsinh(ry) - lsd + logcosh_rq + np.log(s2))   # = 2 Rs cosh(rq) s2, overflow-free
        gt = -np.sign(np.sin(Dl)) * np.exp(logsinh(ry) - lsd + np.log(np.abs(np.sin(Dl))))   # = -Rs sin(Dl), overflow-free
    small = ~(d > 1e-300)
    gr = np.where(small, 0.0, gr); gt = np.where(small, 0.0, gt); d = np.where(small, 0.0, d)
    return d, gr, gt

def objective(rq, tq, ry, ty, alpha):
    d, _, _ = geom(rq, tq, ry, ty)
    return np.mean(d ** alpha, axis=1)

def expmap(rq, tq, vr, vt):
    """exp_q(v) for q = (rq, tq) in polar coordinates about o and v = (vr, vt) in the frame (e_r, e_t)."""
    s = np.hypot(vr, vt)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        cr = np.where(s > 0, vr / s, 0.0); ct = np.where(s > 0, vt / s, 0.0)
        # (a) hyperboloid coordinates, accurate for moderate rq
        ch, sh = np.cosh(s), np.sinh(s)
        x1 = ch * np.sinh(rq) * np.cos(tq) + sh * (cr * np.cosh(rq) * np.cos(tq) - ct * np.sin(tq))
        x2 = ch * np.sinh(rq) * np.sin(tq) + sh * (cr * np.cosh(rq) * np.sin(tq) + ct * np.cos(tq))
        r_a, t_a = np.arcsinh(np.hypot(x1, x2)), np.arctan2(x2, x1)
        # (b) overflow-free hyperbolic trigonometry, used for rq >= 300
        L = np.logaddexp(2 * logsinh(0.5 * np.abs(rq - s)), logsinh(rq) + logsinh(s) + np.log(0.5 * (1 + cr)))
        r_b = np.where(L > 40, L + 2 * LOG2, 2 * np.arcsinh(np.exp(np.minimum(L, 40) / 2)))
        u = np.maximum(0.5 * (s + rq - r_b), 0.0); w = np.maximum(0.5 * (s - rq + r_b), 0.0)
        sin2 = np.exp(logsinh(u) + logsinh(w) - logsinh(rq) - logsinh(r_b))      # sin^2(angle at o / 2)
        t_b = tq + np.sign(ct) * 2 * np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))
    big = rq >= 300
    return np.where(big, r_b, r_a), np.where(big, t_b, t_a)

def frechet_mean(ry, ty, alpha, maxit=1000, tol=1e-13):
    """Batched damped Newton; ry, ty have shape (R, n). Returns r, theta, info."""
    R, n = ry.shape
    # initial value: coordinatewise median of log_o(Y_i) (any start works: the problem is convex)
    x0 = np.median(ry * np.cos(ty), axis=1); y0 = np.median(ry * np.sin(ty), axis=1)
    rq, tq = np.hypot(x0, y0), np.arctan2(y0, x0)
    active = np.ones(R, bool); iters = np.zeros(R, int)
    f = objective(rq[:, None], tq[:, None], ry, ty, alpha)
    for it in range(maxit):
        idx = np.where(active)[0]
        if idx.size == 0: break
        r_, t_ = rq[idx, None], tq[idx, None]
        d, gr, gt = geom(r_, t_, ry[idx], ty[idx])
        w1 = alpha * d ** (alpha - 1)
        Gr, Gt = np.mean(w1 * gr, 1), np.mean(w1 * gt, 1)
        dc = np.maximum(d, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            dcoth = np.where(dc < 1e-6, 1.0, dc / np.tanh(dc))
        w2 = alpha * dc ** (alpha - 2)
        Hrr = np.mean(w2 * ((alpha - 1) * gr * gr + dcoth * (1 - gr * gr)), 1)
        Htt = np.mean(w2 * ((alpha - 1) * gt * gt + dcoth * (1 - gt * gt)), 1)
        Hrt = np.mean(w2 * ((alpha - 1) * gr * gt - dcoth * gr * gt), 1)
        det = Hrr * Htt - Hrt ** 2
        vr = -(Htt * Gr - Hrt * Gt) / det; vt = -(-Hrt * Gr + Hrr * Gt) / det
        dec = -(Gr * vr + Gt * vt)                    # Newton decrement^2 (>= 0)
        scale = np.mean(w1, 1) ** 2 / np.maximum(Hrr + Htt, 1e-300)
        done = dec <= tol * scale
        s = np.hypot(vr, vt); cap = np.minimum(1.0, 5.0 / np.maximum(s, 1e-300))
        vr, vt = vr * cap, vt * cap
        # backtracking line search (Armijo)
        t = np.ones(idx.size); acc = done.copy()
        f0 = f[idx]; rn, tn = r_[:, 0].copy(), t_[:, 0].copy()
        for _ in range(40):
            todo = np.where(~acc)[0]
            if todo.size == 0: break
            rr, tt = expmap(r_[todo, 0], t_[todo, 0], t[todo] * vr[todo], t[todo] * vt[todo])
            fn = objective(rr[:, None], tt[:, None], ry[idx[todo]], ty[idx[todo]], alpha)
            ok = fn <= f0[todo] - 1e-4 * t[todo] * dec[todo] * cap[todo] + 1e-15 * np.abs(f0[todo])
            rn[todo[ok]], tn[todo[ok]] = rr[ok], tt[ok]; f0[todo[ok]] = fn[ok]
            acc[todo[ok]] = True; t[todo[~ok]] *= 0.5
        rq[idx], tq[idx], f[idx] = rn, tn, f0
        iters[idx] += 1
        active[idx[done]] = False
    return rq, tq, dict(max_iter=int(iters.max()), mean_iter=float(iters.mean()),
                        not_converged=int(active.sum()), nc_idx=np.where(active)[0].tolist())

def error_bound(ry, ty, rq, tq, alpha):
    """|grad F| / lambda_min(Hess F) at q = (rq, tq) for F(q) = sum_i d(Y_i, q)^alpha; ry, ty of shape (n,)."""
    a = alpha
    d, gr, gt = geom(np.array([[rq]]), np.array([[tq]]), ry[None], ty[None])
    w1 = a * d ** (a - 1)
    G = np.hypot(np.sum(w1 * gr), np.sum(w1 * gt))
    dc = np.maximum(d, 1e-300)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        dcoth = np.where(dc < 1e-6, 1.0, dc / np.tanh(dc)); w2 = a * dc ** (a - 2)
    Hrr = np.sum(w2 * ((a - 1) * gr * gr + dcoth * (1 - gr * gr)))
    Htt = np.sum(w2 * ((a - 1) * gt * gt + dcoth * (1 - gt * gt)))
    Hrt = np.sum(w2 * ((a - 1) * gr * gt - dcoth * gr * gt))
    lmin = 0.5 * (Hrr + Htt) - np.sqrt(0.25 * (Hrr - Htt) ** 2 + Hrt ** 2)
    return G / lmin if lmin > 0 else np.inf

def sample(rng, R, n, nu=1.5):
    Z = rng.standard_normal((R, n, 2)); W = rng.chisquare(nu, (R, n, 1))
    V = Z / np.sqrt(W / nu)
    return np.hypot(V[..., 0], V[..., 1]), np.arctan2(V[..., 1], V[..., 0])

if __name__ == '__main__':
    alphas = [float(a) for a in sys.argv[1].split(',')]
    ns = [int(x) for x in sys.argv[2].split(',')]
    R = int(sys.argv[3]); out = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20261031
    arrays, info = {}, {}
    for n in ns:
        rng = np.random.default_rng([seed, n])        # independent, reproducible stream per n
        chunk = 2000 if n <= 64 else (1000 if n <= 256 else 500)
        errs = {a: [] for a in alphas}
        inf_n = {a: dict(max_iter=0, not_converged=0, recomputed_high_precision=0) for a in alphas}
        t0 = time.time()
        for c in range(0, R, chunk):
            ry, ty = sample(rng, min(chunk, R - c), n)
            for a in alphas:
                if n == 1:                                 # m_1 = Y_1 exactly
                    errs[a].append(ry[:, 0]); continue
                r, t, inf = frechet_mean(ry, ty, a)
                for i in inf['nc_idx']:                    # rare: float64 iteration did not converge
                    if error_bound(ry[i], ty[i], r[i], t[i], a) > 1e-8 * max(r[i], 1e-300):
                        from mp_newton import frechet_mean_mp
                        r[i] = frechet_mean_mp(ry[i], ty[i], a, start=(r[i], t[i]))[0]
                        inf_n[a]['recomputed_high_precision'] += 1
                errs[a].append(r)
                inf_n[a]['max_iter'] = max(inf_n[a]['max_iter'], inf['max_iter'])
                inf_n[a]['not_converged'] += inf['not_converged']
        for a in alphas:
            arrays[f'err_{a:g}_{n}'] = np.concatenate(errs[a]).astype(np.float32)
            info[f'{a:g}_{n}'] = inf_n[a]
        print(f'n={n} done in {time.time()-t0:.1f}s', flush=True)
        np.savez(out, info=json.dumps(info), **arrays)
