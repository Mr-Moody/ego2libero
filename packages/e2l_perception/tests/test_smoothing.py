import numpy as np

from e2l_perception.smoothing import rts_smooth, smooth_tracks


def test_rts_matches_batch_least_squares():
    """For a linear-Gaussian random walk the RTS smoother equals the batch MAP solution."""
    rng = np.random.default_rng(0)
    N, q, r, m0, P0 = 30, 0.1, 0.5, 0.0, 2.0
    z = np.cumsum(rng.normal(scale=q**0.5, size=N)) + rng.normal(scale=r**0.5, size=N)
    F = np.ones((N, 1, 1))
    xf, Pf, xp, Pp = [], [], [], []
    x, P = np.array([m0]), np.array([[P0]])
    for k in range(N):
        if k > 0:
            x, P = F[k] @ x, F[k] @ P @ F[k].T + q
        xp.append(x.copy())
        Pp.append(P.copy())
        gain = P / (P + r)
        x, P = x + gain[:, 0] * (z[k] - x), (1 - gain) * P
        xf.append(x.copy())
        Pf.append(P.copy())
    xs, Ps = rts_smooth(np.array(xf), np.array(Pf), np.array(xp), np.array(Pp), F)

    A, b = np.zeros((N, N)), np.zeros(N)
    A[0, 0], b[0] = 1 / P0, m0 / P0
    A[np.arange(N), np.arange(N)] += 1 / r
    b += z / r
    for k in range(1, N):
        A[k, k] += 1 / q
        A[k - 1, k - 1] += 1 / q
        A[k, k - 1] -= 1 / q
        A[k - 1, k] -= 1 / q
    np.testing.assert_allclose(xs[:, 0], np.linalg.solve(A, b), atol=1e-9)
    np.testing.assert_allclose(Ps[:, 0, 0], np.diag(np.linalg.inv(A)), atol=1e-9)


def test_rts_accepts_constant_F_and_stacked_tracks():
    N, n, M = 5, 2, 3
    rng = np.random.default_rng(1)
    P = np.tile(np.eye(n), (N, 1, 1))
    x = rng.normal(size=(N, n, M))
    xs, Ps = rts_smooth(x, P, x, P + np.eye(n), np.eye(n))
    assert xs.shape == (N, n, M) and Ps.shape == (N, n, n)
    np.testing.assert_allclose(xs[-1], x[-1])


def _line(N: int = 90, dt: float = 1 / 30) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N) * dt
    truth = np.stack([0.1 + 0.2 * t, -0.05 * t, 0.3 + 0 * t], axis=1)  # (N, 3) metres
    return t, truth


def test_smooth_tracks_reduces_noise_and_fills_gaps():
    t, truth = _line()
    rng = np.random.default_rng(2)
    z = truth + rng.normal(scale=0.01, size=truth.shape)
    valid = np.ones(len(t), bool)
    valid[40:45] = False
    z[~valid] = np.nan
    out = smooth_tracks(z, valid, dt=1 / 30, accel_noise=1.0, meas_std=0.01)
    assert out.shape == z.shape and np.isfinite(out).all()
    raw_rmse = np.sqrt(np.nanmean((z - truth) ** 2))
    rmse = np.sqrt(np.mean((out - truth) ** 2))
    assert rmse < 0.6 * raw_rmse
    # Gap error of the optimal (MAP) estimate is ~0.012 median over seeds; 0.02 is ~p97.
    assert np.abs(out[40:45] - truth[40:45]).max() < 0.02


def test_smooth_tracks_leaves_leading_invalid_rows_and_handles_all_invalid():
    t, truth = _line(20)
    valid = np.ones(20, bool)
    valid[:3] = False
    z = truth.copy()
    z[:3] = np.nan
    out = smooth_tracks(z, valid, dt=1 / 30, accel_noise=1.0, meas_std=0.01)
    assert np.isnan(out[:3]).all() and np.isfinite(out[3:]).all()
    none = smooth_tracks(np.full((5, 2), np.nan), np.zeros(5, bool), 1 / 30, 1.0, 0.01)
    assert np.isnan(none).all()
