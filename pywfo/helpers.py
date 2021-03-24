import numpy as np


def get_dummy_mos(num, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mos = np.random.rand(num, num)
    mos, _ = np.linalg.qr(mos, mode="complete")
    return mos.T


def perturb_mat(mat, scale=2e-1):
    """Add (small) random perturbations to the given matrix.
    This may simulate a small change in the MO-coefficients at a different
    geometry."""
    return mat + np.random.rand(*mat.shape) * scale
