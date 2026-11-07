#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "statsmodels",
#   "numpy",
#   "scipy",
# ]
# ///
#
# Generates the ARMA fixtures under tests/.
#
# Each fixture is a JSON object with a `header` (true orders and window size)
# and a `body` array with one entry per observation. Each body entry holds the
# observation `x`, its delta BIC `dbic` (null when no window fit exists), and
# the window's CSS `phi`/`theta` estimates (empty when none). The delta BIC is
# computed from the *conditional* sum of squares (the same residual recursion
# `arma()` uses) rather than statsmodels' exact-likelihood BIC, so it can be
# checked tightly against the Rust `Fit`.

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from scipy.optimize import least_squares
from scipy.signal import lfilter

TEST_DIR = Path(__file__).resolve().parent

STABILITY_MARGIN = 1.1
MIN_ROOT_DIST = 0.3


def run():
    # Eight samples of every (ar_order, ma_order) parameter set.
    for old in TEST_DIR.glob("arma*.json"):
        old.unlink()

    seed = 0
    for ar_order, ma_order in [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]:
        for _ in range(8):
            seed += 1
            testcase(ar_order, ma_order, 1010, 1000, 1, seed)


def draw_params(ar_order, ma_order, rng):
    while True:
        arparams = rng.uniform(-0.9, 0.9, size=ar_order)
        maparams = rng.uniform(-0.9, 0.9, size=ma_order)
        ap = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
        if np.any(np.abs(ap.arroots) <= STABILITY_MARGIN):
            continue
        if np.any(np.abs(ap.maroots) <= STABILITY_MARGIN):
            continue

        if ap.arroots.size > 0 and ap.maroots.size > 0:
            ar_inv = 1.0 / ap.arroots
            ma_inv = 1.0 / ap.maroots
            dist = np.abs(ar_inv[:, None] - ma_inv[None, :])
            if np.min(dist) < MIN_ROOT_DIST:
                continue

        return arparams, maparams


def delta_bic(x, phi, theta):
    """CSS fit of x, starting at (phi, theta).

    Uses the same residuals as `arma()`: mean removed, residuals from index p,
    MA terms min(q, t - p). Returns (phi, theta, dbic) at the CSS minimum.
    """
    p, q = len(phi), len(theta)
    y = x - x.mean()

    def residuals(beta):
        # AR part: w_t = y_t - sum phi_i * y_{t-i}, for t = p ... n-1
        w = y[p:].copy()
        for i in range(p):
            w -= beta[i] * y[p - 1 - i : len(y) - 1 - i]
        # MA part: e_t + sum theta_k * e_{t-k} = w_t, with e = 0 before index p
        return lfilter([1.0], np.r_[1.0, beta[p:]], w)

    r = least_squares(
        residuals,
        np.r_[phi, theta].astype(float),
        method="lm",
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    m = len(y) - p
    s = float(r.fun @ r.fun)
    s0 = float(y[p:] @ y[p:])
    dbic = m * np.log(s0 / s) - (p + q) * np.log(m)
    return r.x[:p], r.x[p:], dbic


def testcase(ar_order, ma_order, num_obs, win_sz, step, seed, burnin=500):
    rng = np.random.default_rng(seed)

    while True:
        arparams, maparams = draw_params(ar_order, ma_order, rng)
        ap = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
        if not (ap.isstationary and ap.isinvertible):
            continue

        x = ap.generate_sample(num_obs, distrvs=rng.standard_normal, burnin=burnin)

        fit = sm.tsa.ARIMA(x, order=(ar_order, 0, ma_order), trend="c").fit()
        if not fit.mle_retvals.get("converged", True):
            continue

        start_phi, start_theta, d = delta_bic(x, fit.arparams, fit.maparams)
        if d > 10:
            dbic = []
            phi = []
            theta = []

            # Start each window from the previous window's CSS result (the
            # first window uses the full-sample result). That skips the
            # per-window statsmodels fit while finding the same minimum.
            for i in range(0, num_obs - win_sz + 1, step):
                p, t, d = delta_bic(x[i : i + win_sz], start_phi, start_theta)
                dbic.append(float(d))
                phi.append([float(v) for v in p])
                theta.append([float(v) for v in t])
                start_phi, start_theta = p, t
            break

    body = []
    for i, xi in enumerate(x):
        if i < len(dbic):
            body.append(
                {
                    "x": float(xi),
                    "dbic": dbic[i],
                    "phi": phi[i],
                    "theta": theta[i],
                }
            )
        else:
            body.append(
                {
                    "x": float(xi),
                    "dbic": None,
                    "phi": [],
                    "theta": [],
                }
            )

    header = {
        "phi": [float(v) for v in arparams],
        "theta": [float(v) for v in maparams],
        "dbic_window": win_sz,
    }

    fnam = TEST_DIR / f"arma{ar_order}{ma_order}-seed{seed}.json"
    with open(fnam, "w") as f:
        json.dump({"header": header, "body": body}, f)
    print(f"{fnam} written")


if __name__ == "__main__":
    run()
