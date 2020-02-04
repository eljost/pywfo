from pysisyphus.calculators.WFOWrapper2 import WFOWrapper2
import numpy as np

from ..main import overlaps


def write_ref_data(a_mo, b_mo, S_AO=None, out_dir="ref"):
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
    np.set_printoptions(suppress=True, precision=6)

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
