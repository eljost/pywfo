import numpy as np


def perturb_mat(mat, scale=2e-1):
    """Add (small) random perturbations to the given matrix.
    This may simulate a small change in the MO-coefficients at a different
    geometry."""
    return mat + np.random.rand(*mat.shape) * scale
