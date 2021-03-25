import pytest

import h5py
import numpy as np

from pywfo.main import overlaps, moovlp, moovlp_expl, moovlp_dots
from pywfo.helpers import perturb_mat, get_dummy_mos


np.set_printoptions(suppress=True, precision=6)


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


def test_dummy_wfoverlaps():
    """Wavefunction overlaps with dummy data."""

    dim_ = 4
    occ = 2
    virt = dim_ - occ
    states = 2

    # Set up dummy MOs
    bra_mos = get_dummy_mos(dim_, 20180325)
    ket_mos, _ = np.linalg.qr(perturb_mat(bra_mos.T), mode="complete")
    ket_mos = ket_mos.T

    print("Bra MOs")
    print(bra_mos)
    print("Ket MOs")
    print(ket_mos)

    # Dummy CI coefficients
    cis_ = np.zeros((2, states, occ, virt))
    # Bra
    cis_[0, 0, 1, 0] = 1
    cis_[0, 1, 1, 1] = 1

    # Ket
    cis_[1, 0, 1, 1] = 1
    cis_[1, 1, 1, 0] = 1

    print("CI coefficients")
    print(cis_)

    bra_ci, ket_ci = cis_

    # Without GS
    ovlps = overlaps(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ, with_gs=False)
    ref_ovlps = np.array(((-0.0237519286, 0.9491140278), (0.9614129472, 0.0272071813)))
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=1e-5)


@pytest.mark.parametrize(
    "ci_thresh, ref_ovlps",
    [
        # (0.5, (-0.000118, 0.751367, -0.463159, -0.000088)),
        # (0.1, (-0.000117, 0.738117, -0.463159, -0.000088)),  # ~ 0.01 sec
        # (1e-2, (-0.000137, 0.790922, -0.463159, -0.000088)),  # ~  0.2 sec
        (7e-2, (-0.000119, 0.761295, -0.464277, -0.000089)),  # ref
        # (5e-3, (-0.000137, 0.795608, -0.465939, -0.000092)),  # ~   1.28 sec
        # (1e-3, (-0.000137, 0.799300, -0.465685, -0.000092)),  # ~  26.6 sec
    ],
)
def test_cytosin_wfoverlaps(ci_thresh, ref_ovlps, this_dir):
    with h5py.File(this_dir / "ref_cytosin/cytosin_overlap_data.h5") as handle:
        mo_coeffs = handle["mo_coeffs"][:]
        ci_coeffs = handle["ci_coeffs"][:]
    print(mo_coeffs.shape)
    print(ci_coeffs.shape)
    # Compare first and third step 0 and 2
    bra = 0
    ket = 2

    bra_mos = mo_coeffs[bra]
    ket_mos = mo_coeffs[ket]

    bra_ci = ci_coeffs[bra]
    ket_ci = ci_coeffs[ket]

    occ = bra_ci[0].shape[0]

    ovlps = overlaps(
        bra_mos, ket_mos, bra_ci, ket_ci, occ, ci_thresh=ci_thresh, ao_ovlps="ket"
    )
    print(ovlps)
    ref_ovlps = np.array(ref_ovlps).reshape(ovlps.shape)
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=2e-3)


def test_h2o2_wfoverlaps(this_dir):
    """H2O2, BP86/def2-SVP TD-DFT with 4 states.

    Second geometry has slightly rotated dihedral.
    """
    with h5py.File(this_dir / "h2o2_overlap_data.h5") as handle:
        mo_coeffs = handle["mo_coeffs"][:]
        ci_coeffs = handle["ci_coeffs"][:]

    bra_mos, ket_mos = mo_coeffs
    bra_ci, ket_ci = ci_coeffs
    occ = bra_ci[0].shape[0]

    ovlps = overlaps(
        bra_mos,
        ket_mos,
        bra_ci,
        ket_ci,
        occ,
        ci_thresh=1e-2,
        ao_ovlps="bra",
        with_gs=True,
    )
    print(ovlps)
