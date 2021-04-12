import numpy as np


def get_dummy_mos(num, seed=None):
    """Dummy molecular orbital matrix, one MO per row.

    Orthogonalizes a matrix of random entries.
    """
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


def get_bra_ket_dummy_mos(num, occ, seed=None):
    virt = num - occ
    states = 2

    # Set up dummy MOs
    bra_mos = get_dummy_mos(num)
    ket_mos, _ = np.linalg.qr(perturb_mat(bra_mos.T), mode="complete")
    ket_mos = ket_mos.T

    return virt, bra_mos, ket_mos


def get_wfow_ref(bra_mos, ket_mos, bra_ci, ket_ci, ref_fn=None, **wfow_kwargs):
    try:
        from pysisyphus.calculators.WFOWrapper import WFOWrapper

        occ, virt = bra_ci.shape[-2:]

        wfow = WFOWrapper(occ, virt, calc_number=1, **wfow_kwargs)
        old_cycle = bra_mos, bra_ci
        new_cycle = ket_mos, ket_ci
        ref_ovlps = wfow.wf_overlap(old_cycle, new_cycle)
        ref_ovlps = ref_ovlps[0]
        ref_ovlps = ref_ovlps[1:, 1:]
    except:
        ref_ovlps = np.load(ref_fn)
    return ref_ovlps
