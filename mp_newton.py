"""High-precision (mpmath) damped Riemannian Newton method for the sample alpha-Frechet mean in H^2, in geodesic
polar coordinates (r, theta) about o, with cancellation-free formulas.

Used by h2sim.py only for the rare replications in which the float64 iteration does not converge.  A point at
distance r from o is resolved transversally to sinh(r) * 10^(-dps); the working precision is therefore chosen
as dps = r / log(10) + 50 and increased (and the iteration continued) if the minimizer turns out to be farther out.
"""
import math
import mpmath as mp


def _dist(rq, tq, ry, ty):
    S = mp.sinh((ry - rq) / 2) ** 2 + mp.sinh(ry) * mp.sinh(rq) * mp.sin((ty - tq) / 2) ** 2   # sinh^2(d/2)
    return 2 * mp.asinh(mp.sqrt(S))


def _geom(rq, tq, ry, ty):
    """d(q, y) and the unit gradient of d(., y) at q in the frame (e_r, e_t)."""
    D = ty - tq
    s2 = mp.sin(D / 2) ** 2
    S = mp.sinh((ry - rq) / 2) ** 2 + mp.sinh(ry) * mp.sinh(rq) * s2
    d = 2 * mp.asinh(mp.sqrt(S))
    if d == 0:
        return d, mp.mpf(0), mp.mpf(0)
    Rs = mp.sinh(ry) / mp.sinh(d)
    gr = -mp.sinh(ry - rq) / mp.sinh(d) + 2 * Rs * mp.cosh(rq) * s2
    gt = -Rs * mp.sin(D)
    return d, gr, gt


def _exp(rq, tq, vr, vt):
    """exp_q(v) for q = (rq, tq) and v = (vr, vt) in the frame (e_r, e_t)."""
    s = mp.sqrt(vr ** 2 + vt ** 2)
    if s == 0:
        return rq, tq
    if rq < mp.mpf(10) ** (-(mp.mp.dps // 2)):
        return s, tq + mp.atan2(vt, vr)
    c2 = (1 + vr / s) / 2                                                   # cos^2(phi/2), phi = angle(v, e_r)
    rn = 2 * mp.asinh(mp.sqrt(mp.sinh((rq - s) / 2) ** 2 + mp.sinh(rq) * mp.sinh(s) * c2))
    if rn == 0:
        return rn, tq
    q = mp.sinh((s + rq - rn) / 2) * mp.sinh((s - rq + rn) / 2) / (mp.sinh(rq) * mp.sinh(rn))
    q = min(max(q, mp.mpf(0)), mp.mpf(1))                                   # sin^2(angle at o / 2)
    return rn, tq + mp.sign(vt) * 2 * mp.asin(mp.sqrt(q))


def _newton(Y, a, rq, tq, maxit):
    rmax = (mp.mp.dps - 30) * mp.log(10)
    tiny = mp.mpf(10) ** (-(mp.mp.dps // 3))
    F = lambda r, t: mp.fsum(_dist(r, t, ry, ty) ** a for ry, ty in Y)
    f = F(rq, tq)
    nv = mp.inf
    for it in range(maxit):
        G = [mp.mpf(0), mp.mpf(0)]
        H = [[mp.mpf(0)] * 2 for _ in range(2)]
        for ry, ty in Y:
            d, gr, gt = _geom(rq, tq, ry, ty)
            if d < tiny:                                                    # q sits on a data point
                continue
            g = (gr, gt)
            w1, w2, dc = a * d ** (a - 1), a * d ** (a - 2), d / mp.tanh(d)
            for i in range(2):
                G[i] += w1 * g[i]
                for j in range(2):
                    H[i][j] += w2 * ((a - 1) * g[i] * g[j] + dc * ((1 if i == j else 0) - g[i] * g[j]))
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        vr = -(H[1][1] * G[0] - H[0][1] * G[1]) / det
        vt = -(-H[1][0] * G[0] + H[0][0] * G[1]) / det
        dec = -(G[0] * vr + G[1] * vt)
        nv = mp.sqrt(vr ** 2 + vt ** 2)
        if nv < mp.mpf(10) ** (-30):
            break
        cap = max(mp.mpf(50), rq)                                           # trust region for far-out iterates
        if nv > cap:
            vr, vt, dec = vr * cap / nv, vt * cap / nv, dec * cap / nv
        step, accepted = mp.mpf(1), False
        while step > mp.mpf(10) ** (-30):                                   # Armijo backtracking
            rn, tn = _exp(rq, tq, step * vr, step * vt)
            fn = F(rn, tn)
            if fn <= f - mp.mpf('1e-4') * step * dec:
                accepted = True
                break
            step /= 2
        if not accepted:                                                    # no further decrease representable
            break
        rq, tq, f = rn, tn, fn
        if rq > rmax:                                                       # beyond the resolvable radius
            break
    return rq, tq, it, nv


def frechet_mean_mp(ry, ty, alpha, start=None, maxit=2000):
    """Sample alpha-Frechet mean of the points (ry[i], ty[i]) (polar coordinates about o).
    start: optional (r, theta) initial value (e.g. the last float64 iterate); default: coordinatewise median
    of log_o(Y_i).  Returns (d(o, m_n), theta(m_n), iterations, length of the last Newton step, dps)."""
    k = len(ry)
    if start is None:
        xs = sorted(float(r) * math.cos(float(t)) for r, t in zip(ry, ty))
        ys = sorted(float(r) * math.sin(float(t)) for r, t in zip(ry, ty))
        med = lambda v: v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2
        start = (math.hypot(med(xs), med(ys)), math.atan2(med(ys), med(xs)))
    r0, t0 = float(start[0]), float(start[1])
    dps, total = int(r0 / math.log(10)) + 50, 0
    while True:
        mp.mp.dps = dps
        Y = [(mp.mpf(float(r)), mp.mpf(float(t))) for r, t in zip(ry, ty)]
        rq, tq, it, nv = _newton(Y, mp.mpf(alpha), mp.mpf(r0), mp.mpf(t0), maxit)
        total += it + 1
        if float(rq) / math.log(10) + 30 < dps:                             # transverse resolution < 1e-30
            return float(rq), float(tq), total, float(nv), dps
        r0, t0 = rq, tq                                                     # continue at higher precision
        dps = int(1.5 * float(rq) / math.log(10)) + 50
