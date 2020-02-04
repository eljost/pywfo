import numpy as np

from main import overlaps


def perturb_mat(mat, scale=2e-1):
    """Add (small) random perturbations to the given matrix."""
    return mat + np.random.rand(*mat.shape)*scale



def test_ovlps():
    np.set_printoptions(suppress=True, precision=6)

    ref_ovlps = np.array((
                    (-0.0237519286, 0.9491140278),
                    (0.9614129472, 0.0272071813))
    )

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


    ovlps = overlaps(bra_mos, ket_mos, bra_ci, ket_ci, occ=occ)
    np.testing.assert_allclose(ovlps, ref_ovlps, atol=1e-5)

