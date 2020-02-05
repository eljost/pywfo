import os
from pathlib import Path
import pytest

import h5py
import numpy as np

from ..main import overlaps, overlaps2


np.set_printoptions(suppress=True, precision=6)

THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def write_ref_data(a_mo, b_mo, S_AO=None, out_dir="ref"):
    from pysisyphus.calculators.WFOWrapper2 import WFOWrapper2
    for fn, mos in zip("a_mo b_mo".split(), (a_mo, b_mo)):
        mo_str = WFOWrapper2.fake_turbo_mos(mos)
        with open(f"{out_dir}/{fn}", "w") as handle:
            handle.write(mo_str)

    if S_AO is not None:
        a_AOs, b_AOs = S_AO.shape
        with io.StringIO() as _:
            np.savetxt(_, S_AO)
            S_AO_str = f"{a_AOs} {b_AOs}\n{_.getvalue()}"
        with open(f"{out_dir}/ao_ovlp", "w") as handle:
            handle.write(S_AO_str)


def perturb_mat(mat, scale=2e-1):
    """Add (small) random perturbations to the given matrix.
    This may simulate a small change in the MO-coefficients at a different
    geometry."""
    return mat + np.random.rand(*mat.shape)*scale


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
    cis_[0,0,1,0] = 1
    cis_[0,1,1,1] = 1

    # Ket
    cis_[1,0,1,1] = 1
    cis_[1,1,1,0] = 1
    
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
    ref_ovlps = np.array((
                    (-0.0237519286, 0.9491140278),
                    (0.9614129472, 0.0272071813))
    )
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
    "ci_thresh, ref_ovlps",
    [
     # (.5, (-0.0001179859,  0.7513668694, -0.4631592028, -0.0000876285)),
     (1e-2, (-0.000137,  0.790922, -0.463159, -0.000088)),  # ~ 4 sec
     # (1e-3, (-0.000137,  0.795608, -0.465939, -0.000092)),  # ~ 5.1 sec, overlap2
     # (1e-3, (-0.000137,  0.799300, -0.465685, -0.000092)),  # ~ 5.1 sec, overlap2
    ]
)
def test_cytosin(ci_thresh, ref_ovlps):
    with h5py.File(THIS_DIR / "ref_cytosin/cytosin_overlap_data.h5") as handle:
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

    ovlps = overlaps2(bra_mos, ket_mos, bra_ci, ket_ci, occ, ci_thresh=ci_thresh,
                    ao_ovlps="ket")
    print(ovlps)
    ref_ovlps = np.array(ref_ovlps).reshape(ovlps.shape)
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=2e-3)
