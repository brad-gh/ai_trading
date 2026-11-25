"""Vectorised Black-Scholes and IV inversion."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq


def bs(S, K, T, r, sig, cp):
    """cp = +1 call, -1 put. Returns intrinsic when T <= 0."""
    S, K, T, r, sig, cp = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, r, sig, cp)))
    intrinsic = np.maximum(cp * (S - K), 0.0)
    live = T > 1e-9
    out = intrinsic.copy()
    if live.any():
        s, k, t, rr, v, c = S[live], K[live], T[live], r[live], np.maximum(sig[live], 1e-4), cp[live]
        sq = v * np.sqrt(t)
        d1 = (np.log(s / k) + (rr + 0.5 * v * v) * t) / sq
        d2 = d1 - sq
        out[live] = c * (s * ndtr(c * d1) - k * np.exp(-rr * t) * ndtr(c * d2))
    return out


def implied_vol(price, S, K, T, r, cp):
    intrinsic = max(cp * (S - K * np.exp(-r * T)), 0.0)
    if price <= intrinsic + 1e-6 or T <= 0:
        return np.nan
    f = lambda v: float(bs(S, K, T, r, v, cp)[0]) - price
    try:
        return brentq(f, 1e-3, 6.0, maxiter=200)
    except ValueError:
        return np.nan


def skew_mult(K, S, T, a, b):
    """IV(K)/IV_ATM as quadratic in normalised moneyness m = ln(K/S)/sqrt(T)."""
    m = np.clip(np.log(K / S) / np.sqrt(np.maximum(T, 1 / 252)), -3.0, 3.0)
    return np.clip(1.0 + a * m + b * m * m, 0.6, 2.5)
