import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import null_space

KAPPA = 0.1
LAMBDA = np.array([[1.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.5, 0.5, 0.0]])
A0 = np.array([0.0, -1.0, 2.0])
A1 = np.array([1.0, 1.0, -2.0])
A2 = np.array([-1.0, 1.0, 0.0])
ALPHA = np.array([0.2, 0.4])
BETA = np.array([0.6, 0.4])
OMEGA0 = 1.0 - ALPHA - BETA


def population_geometry():
    # E[Y_t Y_t']
    sigma_y = LAMBDA[:, :2] @ LAMBDA[:, :2].T + KAPPA * np.eye(3)

    # Exact fourth-moment block E[Y_i^2 Y_j^2].
    # Independent components: f1, f2, e1, e2, e3.
    variances = np.array([1.0, 1.0, 0.1, 0.1, 0.1])
    cumulant4 = np.array([6.0 / 7.0, 24.0, 0.0, 0.0, 0.0])
    coeff = np.array([[1.0, 0.0, 1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0, 1.0, 0.0],
                      [0.5, 0.5, 0.0, 0.0, 1.0]])

    def cross4(a, b):
        va = np.sum(a * a * variances)
        vb = np.sum(b * b * variances)
        cab = np.sum(a * b * variances)
        return va * vb + 2.0 * cab * cab + np.sum(cumulant4 * a * a * b * b)

    m4 = np.array([[cross4(coeff[i], coeff[j]) for j in range(3)] for i in range(3)])
    ezz = np.zeros((6, 6))
    ezz[:3, :3] = sigma_y
    ezz[3:, 3:] = m4

    # At theta0, Var(u0)=1/2 and u0 is independent of z_t.
    omega = 0.5 * ezz
    w = np.linalg.inv(omega)

    d = np.r_[np.zeros(3), [11/10, 21/10, 3/5]]
    c1 = np.r_[np.zeros(3), [129/70, 199/70, 11/14]]
    c2 = np.r_[np.zeros(3), [11/10, 157/10, 4.0]]

    def residualize(c):
        return c - d * ((d @ w @ c) / (d @ w @ d))

    c = np.column_stack([residualize(c1), residualize(c2)])
    g = c.T @ w @ c
    det_g = np.linalg.det(g)
    rho = g[0, 1] / math.sqrt(g[0, 0] * g[1, 1])
    p_empty = 0.25 + math.asin(rho) / (2 * math.pi)
    p_both = 0.25 - math.asin(rho) / (2 * math.pi)

    return {
        "Ezz": ezz, "W": w, "d": d, "C": c, "G": g,
        "det_G": det_g, "rho_G": rho,
        "p_empty": p_empty, "p_1": 0.25, "p_2": 0.25, "p_both": p_both,
    }


def active_key(y, tol=1e-12):
    return tuple(np.flatnonzero(np.asarray(y) > tol).tolist())


def branch_spaces(geom):
    """Euclidean orthonormal bases for N_A={w:d'w=0,c_j'w=0,j in A}."""
    d = geom["d"]
    # Recover unprojected c_j from the exact published-design values.
    c_raw = np.column_stack([
        np.r_[np.zeros(3), [129/70, 199/70, 11/14]],
        np.r_[np.zeros(3), [11/10, 157/10, 4.0]],
    ])
    spaces = {}
    for A in [(0,), (1,), (0, 1)]:
        constraints = np.column_stack([d] + [c_raw[:, j] for j in A])
        U = null_space(constraints.T)
        # Numerical verification of the defining restrictions.
        err = np.max(np.abs(constraints.T @ U)) if U.size else 0.0
        if err > 1e-11:
            raise RuntimeError(f"branch-space construction failed for A={A}: {err}")
        spaces[A] = U
    return spaces


def leading_state(out_i, n, geom, ylead):
    """Population-projected leading residual and its dual vector."""
    w = geom["W"]
    d0 = geom["d"]
    g0 = out_i["g0"]
    # Z_perp = sqrt(n) * Pi_d gbar(theta0), using the population regular direction.
    zp = math.sqrt(n) * (g0 - d0 * ((d0 @ w @ g0) / (d0 @ w @ d0)))
    rstar = zp + geom["C"] @ ylead
    wstar = w @ rstar
    return zp, rstar, wstar


def solve_nnqp(h, g):
    invg = np.linalg.inv(g)
    candidates = [np.zeros(2)]
    y1 = max(-h[0] / g[0, 0], 0.0)
    y2 = max(-h[1] / g[1, 1], 0.0)
    candidates += [np.array([y1, 0.0]), np.array([0.0, y2])]
    yb = -invg @ h
    if np.all(yb >= 0.0):
        candidates.append(yb)
    vals = [y @ g @ y + 2.0 * h @ y for y in candidates]
    return candidates[int(np.argmin(vals))]


def simulate_coefficients(replications, checkpoints, burn, seed):
    rng = np.random.default_rng(seed)
    maxn = max(checkpoints)
    h = np.ones((replications, 2))
    f = np.sqrt(h) * rng.standard_normal((replications, 2))
    e = math.sqrt(KAPPA) * rng.standard_normal((replications, 3))
    y = f @ LAMBDA[:, :2].T + e
    h = OMEGA0 + ALPHA * f * f + BETA * h

    for _ in range(burn):
        f = np.sqrt(h) * rng.standard_normal((replications, 2))
        e = math.sqrt(KAPPA) * rng.standard_normal((replications, 3))
        y = f @ LAMBDA[:, :2].T + e
        h = OMEGA0 + ALPHA * f * f + BETA * h

    names = ["d", "g0", "b1", "b2", "q11", "q22", "q12"]
    acc = {nm: np.zeros((replications, 6)) for nm in names}
    out = {}

    for t in range(1, maxn + 1):
        z = np.concatenate([y, y * y], axis=1)
        f_next = np.sqrt(h) * rng.standard_normal((replications, 2))
        e_next = math.sqrt(KAPPA) * rng.standard_normal((replications, 3))
        y_next = f_next @ LAMBDA[:, :2].T + e_next
        h_next = OMEGA0 + ALPHA * f_next * f_next + BETA * h

        r0 = y_next @ A0
        r1 = y_next @ A1
        r2 = y_next @ A2
        u0 = r0 * r0 - KAPPA * (A0 @ A0)
        psi1 = 2 * r0 * r1 - 2 * KAPPA * (A0 @ A1)
        psi2 = 2 * r0 * r2 - 2 * KAPPA * (A0 @ A2)
        phi11 = r1 * r1 - KAPPA * (A1 @ A1)
        phi22 = r2 * r2 - KAPPA * (A2 @ A2)
        phi12 = 2 * (r1 * r2 - KAPPA * (A1 @ A2))
        vals = [np.ones(replications), u0, psi1, psi2, phi11, phi22, phi12]
        for nm, val in zip(names, vals):
            acc[nm] += z * val[:, None]

        if t in checkpoints:
            out[t] = {nm: arr / t for nm, arr in acc.items()}
        y, h = y_next, h_next

    return out


def oracle_estimate_one(coef, n, geom, bound=1.5):
    w, c = geom["W"], geom["C"]
    d = coef["d"]
    g0, b1, b2 = coef["g0"], coef["b1"], coef["b2"]
    q11, q22, q12 = coef["q11"], coef["q22"], coef["q12"]
    wd = w @ d
    hmat = w - np.outer(wd, wd) / (d @ wd)

    def fun(x):
        x1, x2 = x
        v = g0 + b1*x1 + b2*x2 + q11*x1*x1 + q22*x2*x2 + q12*x1*x2
        return float(v @ hmat @ v)

    def jac(x):
        x1, x2 = x
        v = g0 + b1*x1 + b2*x2 + q11*x1*x1 + q22*x2*x2 + q12*x1*x2
        hv = hmat @ v
        dv1 = b1 + 2*q11*x1 + q12*x2
        dv2 = b2 + 2*q22*x2 + q12*x1
        return 2*np.array([dv1 @ hv, dv2 @ hv])

    hlead = math.sqrt(n) * (c.T @ (w @ g0))
    ylead = solve_nnqp(hlead, geom["G"])
    mag = np.sqrt(np.maximum(ylead, 0.0)) * n**(-0.25)
    starts = [np.zeros(2)]
    s1 = [-1, 1] if mag[0] > 1e-12 else [0]
    s2 = [-1, 1] if mag[1] > 1e-12 else [0]
    for a in s1:
        for b in s2:
            starts.append(np.array([a*mag[0], b*mag[1]]))
    starts += [np.array([sx*0.25, sy*0.25]) for sx in (-1, 1) for sy in (-1, 1)]

    best = None
    for st in starts:
        res = minimize(fun, st, jac=jac, method="L-BFGS-B",
                       bounds=[(-bound, bound), (-bound, bound)],
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, ylead, hlead


def branch_score_accuracy(out_n, n, xhat, ylead, geom):
    """Branch accuracy using only theorem-relevant projected scores U_A'B_j.

    Also returns diagnostics showing (i) dual residuals satisfy the active
    orthogonality conditions and (ii) the projected scalar score equals the
    direct raw-vector inner product up to numerical precision.
    """
    spaces = branch_spaces(geom)
    d0 = geom["d"]
    c_raw = np.column_stack([
        np.r_[np.zeros(3), [129/70, 199/70, 11/14]],
        np.r_[np.zeros(3), [11/10, 157/10, 4.0]],
    ])
    pred = np.zeros_like(xhat)
    margins = np.full_like(xhat, np.nan, dtype=float)
    active = ylead > 1e-12
    max_orth_err = 0.0
    max_projection_identity_err = 0.0

    for i in range(len(xhat)):
        A = active_key(ylead[i])
        if not A:
            continue
        U = spaces[A]
        out_i = {k: v[i] for k, v in out_n.items()}
        _, _, wstar = leading_state(out_i, n, geom, ylead[i])

        # KKT/regular-coordinate orthogonality diagnostics.
        constraints = np.column_stack([d0] + [c_raw[:, j] for j in A])
        max_orth_err = max(max_orth_err, float(np.max(np.abs(constraints.T @ wstar))))

        omega_A = U.T @ wstar
        for j in A:
            Bj = math.sqrt(n) * out_n["b1" if j == 0 else "b2"][i]
            projected_B = U.T @ Bj
            score_proj = float(projected_B @ omega_A)
            score_direct = float(Bj @ wstar)
            max_projection_identity_err = max(
                max_projection_identity_err, abs(score_proj - score_direct)
            )
            margins[i, j] = score_proj
            pred[i, j] = -np.sign(score_proj)

    actual = np.sign(xhat)
    acc = []
    for j in range(2):
        m = active[:, j]
        acc.append(np.mean(pred[m, j] == actual[m, j]) if np.any(m) else np.nan)
    both = np.all(active, axis=1)
    joint = np.mean(np.all(pred[both] == actual[both], axis=1)) if np.any(both) else np.nan

    # Margin-stratified accuracy among active observations. The cuts are
    # empirical tertiles because score scales differ by direction.
    margin_stats = {}
    for j in range(2):
        m = active[:, j] & np.isfinite(margins[:, j])
        vals = np.abs(margins[m, j])
        if len(vals) >= 3:
            q1, q2 = np.quantile(vals, [1/3, 2/3])
            bins = [vals <= q1, (vals > q1) & (vals <= q2), vals > q2]
            idx = np.flatnonzero(m)
            for label, bm in zip(["low", "mid", "high"], bins):
                sel = idx[bm]
                margin_stats[f"BSA{j+1}_{label}"] = (
                    float(np.mean(pred[sel, j] == actual[sel, j])) if len(sel) else np.nan
                )
        else:
            for label in ["low", "mid", "high"]:
                margin_stats[f"BSA{j+1}_{label}"] = np.nan

    return {
        "BSA1_projected": acc[0],
        "BSA2_projected": acc[1],
        "BSA_joint_both_projected": joint,
        "max_active_orthogonality_error": max_orth_err,
        "max_projection_identity_error": max_projection_identity_err,
        **margin_stats,
    }


def run(replications=1000, checkpoints=(500,1000,2000,5000,10000), burn=2000, seed=20260917):
    geom = population_geometry()
    print("G=\n", geom["G"])
    print("det(G)=", geom["det_G"])
    print("rho_G=", geom["rho_G"])
    print("active probs=", geom["p_empty"], geom["p_1"], geom["p_2"], geom["p_both"])

    t0 = time.time()
    outs = simulate_coefficients(replications, checkpoints, burn, seed)
    rows = []
    for n in checkpoints:
        xhat = np.zeros((replications, 2))
        ylead = np.zeros((replications, 2))
        hlead = np.zeros((replications, 2))
        for i in range(replications):
            coef = {k: v[i] for k, v in outs[n].items()}
            xhat[i], ylead[i], hlead[i] = oracle_estimate_one(coef, n, geom)

        code = (ylead[:,0] > 1e-12).astype(int) + 2*(ylead[:,1] > 1e-12).astype(int)
        freq = np.bincount(code, minlength=4) / replications
        branch_diag = branch_score_accuracy(outs[n], n, xhat, ylead, geom)
        rows.append({
            "n": n,
            "MAD_x1": np.median(np.abs(xhat[:,0])),
            "MAD_x2": np.median(np.abs(xhat[:,1])),
            "MAD_scaled_x1": np.median(np.abs(n**0.25*xhat[:,0])),
            "MAD_scaled_x2": np.median(np.abs(n**0.25*xhat[:,1])),
            "P_empty_leading": freq[0],
            "P_A1_leading": freq[1],
            "P_A2_leading": freq[2],
            "P_both_leading": freq[3],
            **branch_diag,
            "mean_q22_Y2sq": outs[n]["q22"][:,4].mean(),
            "median_q22_Y2sq": np.median(outs[n]["q22"][:,4]),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"elapsed={time.time()-t0:.2f}s")
    return geom, df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replications", type=int, default=1000)
    p.add_argument("--burn", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--out", type=str, default="common_ch_mc_results.csv")
    args = p.parse_args()
    geom, df = run(replications=args.replications, burn=args.burn, seed=args.seed)
    df.to_csv(args.out, index=False)
    print("saved", Path(args.out).resolve())
