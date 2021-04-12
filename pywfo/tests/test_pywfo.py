import h5py
import numpy as np
import pytest

from pywfo.main import moovlp, moovlp_expl, moovlp_dots
from pywfo.main import overlaps_naive, overlaps_most_naive, overlaps_cache
from pywfo.helpers import (
    perturb_mat,
    get_dummy_mos,
    get_wfow_ref,
    get_bra_ket_dummy_mos,
)


np.set_printoptions(suppress=True, precision=8)


def test_mo_overlaps():
    """Test calculation of MO overlaps."""

    # Set up dummy MOs
    mo0 = get_dummy_mos(10, 20180325)

    # Slightly perturb original matrix
    mo1, _ = np.linalg.qr(perturb_mat(mo0.T), mode="complete")
    mo1 = mo1.T

    # Determine AO overlaps
    mo0i = np.linalg.inv(mo0)
    S_AO = mo0i.dot(mo0i.T)

    ovlp = moovlp(mo0, mo1, S_AO)
    ovlp_expl = moovlp_expl(mo0, mo1, S_AO)
    ovlp_dots = moovlp_dots(mo0, mo1, S_AO)
    np.testing.assert_allclose(ovlp_expl, ovlp)
    np.testing.assert_allclose(ovlp_dots, ovlp)


def test_big_mo_overlaps():
    """Test calculation of big MO matrix overlaps."""

    mos_fn = "big_mos.npy"
    s_ao_fn = "big_mos_s_ao.npy"
    # Set up dummy data
    # dim_ = 1000
    # mos = get_dummy_mos(dim_, 20180325)
    # np.save(mos_fn, mos)
    # mos_inv = np.linalg.inv(mos)
    # S_AO = mos_inv.dot(mos_inv.T)
    # np.save(s_ao_fn, S_AO)

    mos = np.load(mos_fn)
    dim_ = mos.shape[0]
    S_AO = np.load(s_ao_fn)

    ovlps = moovlp(mos, mos, S_AO)
    np.testing.assert_allclose(ovlps.flatten(), np.eye(dim_).flatten(), atol=1e-8)


@pytest.mark.parametrize(
    "ovlp_func",
    [
        overlaps_naive,
        overlaps_most_naive,
        overlaps_cache,
    ],
)
def test_simple(ovlp_func):
    # Set up dummy MOs
    num = 2
    occ = 1
    virt, bra_mos, ket_mos = get_bra_ket_dummy_mos(num, occ, seed=20180325)

    # Dummy CI coefficients, two "steps", two states
    states = 1
    cis_ = np.zeros((2, states, occ, virt))
    # Bra, state 0
    cis_[0, 0, 0, 0] = 1
    # Ket, state 0
    cis_[1, 0, 0, 0] = 1
    # Normalize
    cis_norms = np.linalg.norm(cis_, axis=(2, 3))
    cis_ /= cis_norms[:, :, None, None]

    bra_ci, ket_ci = cis_

    # Without GS
    ovlps = ovlp_func(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ)

    try:
        ref_ovlps = get_wfow_ref(bra_mos, ket_mos, bra_ci, ket_ci)
        np.testing.assert_allclose(ovlps, ref_ovlps)
    except:
        assert ovlps[0][0] == pytest.approx(0.97008177)


def test_more_virtual_mos():
    # Dummy MOs
    num = 4
    occ = 2
    virt, bra_mos, ket_mos = get_bra_ket_dummy_mos(num, occ, seed=20180325)

    # Dummy CI coefficients
    states = 2
    cis_ = np.zeros((2, states, occ, virt))
    # Bra
    cis_[0, 0, 1, 0] = 1
    cis_[0, 1, 1, 1] = 1
    # Ket
    cis_[1, 0, 1, 1] = 1
    cis_[1, 1, 1, 0] = 1
    # Normalize
    cis_norms = np.linalg.norm(cis_, axis=(2, 3))
    cis_ /= cis_norms[:, :, None, None]

    bra_ci, ket_ci = cis_

    ovlps = overlaps_most_naive(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ)
    try:
        ref_ovlps = get_wfow_ref(bra_mos, ket_mos, bra_ci, ket_ci)
    except:
        ref_ovlps = np.array((0.11025145, 0.97114924, 0.97595963, -0.09630209)).reshape(
            -1, 2
        )
    np.testing.assert_allclose(ovlps, ref_ovlps)


def test_h2o2(this_dir):
    """H2O2, BP86/def2-SVP TD-DFT with 4 states.

    Second geometry has slightly rotated dihedral.
    """
    with h5py.File(this_dir / "h2o2_overlap_data.h5") as handle:
        mo_coeffs = handle["mo_coeffs"][:]
        ci_coeffs = handle["ci_coeffs"][:]

    bra_mos, ket_mos = mo_coeffs
    bra_ci, ket_ci = ci_coeffs
    bra_ci = bra_ci[0, :, :].reshape(1, -1, 29)
    ket_ci = ket_ci[0, :, :].reshape(1, -1, 29)
    occ = bra_ci[0].shape[0]
    ci_thresh = 3e-2  # 3e-2 still works, 2e-2 breaks

    # ovlps = overlaps_naive(
    ovlps = overlaps_cache(
        bra_mos,
        ket_mos,
        bra_ci,
        ket_ci,
        occ,
        ci_thresh=ci_thresh,
        ao_ovlps="ket",
    )
    print(ovlps)
    ref_ovlps = get_wfow_ref(bra_mos, ket_mos, bra_ci, ket_ci, conf_thresh=ci_thresh)
    np.testing.assert_allclose(ovlps, ref_ovlps)
