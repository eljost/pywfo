import pytest

import h5py
import numpy as np

from pywfo.main import overlaps, overlaps2
from pywfo.helpers import perturb_mat


np.set_printoptions(suppress=True, precision=6)


def test_ovlps():
    # Construct dummy MOs
    np.random.seed(20180325)
    dim_ = 4
    occ = 2
    states = 2
    virt = dim_ - occ
    _ = np.random.rand(dim_, dim_)
    bra_mos, _ = np.linalg.qr(_, mode="complete")
    # MOs are given per row
    bra_mos = bra_mos.T
    ket_mos, _ = np.linalg.qr(perturb_mat(bra_mos.T), mode="complete")
    ket_mos = ket_mos.T
    # ket_mos = bra_mos

    print("Bra MOs")
    print(bra_mos)
    print("Ket MOs")
    print(ket_mos)

    # CI coefficients
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

    # c = np.zeros((2,2))
    # c[1,0] = .7
    # c[1,1] = .6
    # ex_ = np.nonzero(c > ci_thresh)
    # _ = get_sd_mo_inds(bra_mos, exc=ex_)

    # bra_mos_inv = np.linalg.inv(bra_mos)
    # S_AO = bra_mos_inv.dot(bra_mos_inv.T)
    # write_ref_data(bra_mos, ket_mos, S_AO)

    # Without GS
    ovlps = overlaps(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ, with_gs=False)
    print(ovlps)
    ref_ovlps = np.array(((-0.0237519286, 0.9491140278), (0.9614129472, 0.0272071813)))
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=1e-5)

    # With GS
    # ovlps = overlaps(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ, with_gs=True)
    # ref_ovlps = np.array((
    # (0.9695194141,  0.0000000000, 0.00000000000),
    # (0.0000000000, -0.0237519286, 0.9491140278),
    # (0.0000000000,  0.9614129472, 0.0272071813))
    # )
    # np.testing.assert_allclose(ovlps, ref_ovlps, atol=1e-5)


@pytest.mark.parametrize(
    "ci_thresh, ref_ovlps, ovlp_func",
    [
        # (.5, (-0.0001179859,  0.7513668694, -0.4631592028, -0.0000876285), "overlap2"),
        # (.1, (-0.000117,  0.738117, -0.463159, -0.000088), "overlap2"),      # ~ 0.01 sec
        # (1e-2, (-0.000137,  0.790922, -0.463159, -0.000088), "overlap"),     # ~  5.4 sec
        # (1e-2, (-0.000137,  0.790922, -0.463159, -0.000088), "overlap2"),    # ~  0.2 sec
        (7e-2, (-0.000119, 0.761295, -0.464277, -0.000089), "overlap2"),  # ref
        # (5e-3, (-0.000137,  0.795608, -0.465939, -0.000092), "overlap"),     # ~  44   sec
        # (5e-3, (-0.000137,  0.795608, -0.465939, -0.000092), "overlap2"),    # ~   1.28 sec
        # (1e-3, (-0.000137,  0.799300, -0.465685, -0.000092), "overlap2"),    # ~  26.6 sec
    ],
)
def test_cytosin(ci_thresh, ref_ovlps, ovlp_func, this_dir):
    ovlp_funcs = {
        "overlap": overlaps,
        "overlap2": overlaps2,
    }
    func = ovlp_funcs[ovlp_func]

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

    ovlps = func(
        bra_mos, ket_mos, bra_ci, ket_ci, occ, ci_thresh=ci_thresh, ao_ovlps="ket"
    )
    print(ovlps)
    ref_ovlps = np.array(ref_ovlps).reshape(ovlps.shape)
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=2e-3)
