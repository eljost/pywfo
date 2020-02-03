# http://dx.doi.org/10.1021/acs.jctc.5b01148

import h5py
import numpy as np

np.set_printoptions(suppress=True, precision=4)


def get_mo_ovlp(shape):
    _ = np.zeros(shape)
    path, string_repr = np.einsum_path("pu,qv,uv->pq", _, _, _,
                                       optimize="optimal")

    def moovlp(mos1, mos2, S_AO=None):
        """Overlap between two sets of mos."""
        if S_AO is None:
            mos1_inv = np.linalg.inv(mos1)
            S_AO = mos1_inv.dot(mos1_inv.T)

        # <phi_p|phi_q> = sum_{u,v} C_{pu} C'_{qv} <X_u|X'_v>
        ovlp = np.einsum("pu,qv,uv->pq", mos1, mos2, S_AO, optimize=path)
        return ovlp
    return moovlp


def perturb_mat(mat, scale=5e-2):
    """Add (small) random perturbations to the given matrix."""
    return mat + np.random.rand(*mat.shape)*scale


def run():
    # 4 MOs, 2 states
    dim_ = 4
    occ = 2
    ci_thresh = .5
    states = 2

    virt = dim_ - occ
    nel = occ
    # Indexes the ground state MOs without any excitation
    mo_mask = np.arange(nel)
    # One row per electron/MO, one column per basis function
    sd_mos_shape = (nel, dim_)

    def get_sd_mos(mos, exc=None):
        """Get MOs to form a Slater determinant for the given excitations."""
        sd_mos = np.zeros(sd_mos_shape)
        mo_inds = mo_mask.copy()
        if exc:
            exc_from, exc_to = exc
        mo_inds[exc_from] = exc_to + occ
        return mos[mo_inds]

    moovlp = get_mo_ovlp((dim_, dim_))

    # Construct dummy MOs
    np.random.seed(20180325)
    _ = np.random.rand(dim_, dim_)
    bra_mos, _ = np.linalg.qr(_, mode="complete")
    # MOs are given per row
    bra_mos = bra_mos.T
    ket_mos, _ = np.linalg.qr(perturb_mat(bra_mos), mode="complete")
    ket_mos = ket_mos.T

    bra_mos_inv = np.linalg.inv(bra_mos)
    S_AO = bra_mos_inv.dot(bra_mos_inv.T)

    # CI coefficients
    cis_ = np.zeros((2, states, occ, virt))
    # Bra
    cis_[0,0,1,0] = 1
    cis_[0,1,1,1] = 1

    # Ket
    cis_[1,0,1,1] = 1
    cis_[1,1,1,0] = 1

    bra_cis, ket_cis = cis_

    # Coefficients above threshold per state
    bra_exc = [np.where(_ > ci_thresh) for _ in bra_cis]
    ket_exc = [np.where(_ > ci_thresh) for _ in ket_cis]

    # MOs that make up the Slater determinants
    bra_sd_mos = [get_sd_mos(bra_mos, exc) for exc in bra_exc]
    ket_sd_mos = [get_sd_mos(ket_mos, exc) for exc in ket_exc]
    import pdb; pdb.set_trace()


    sd_mos = get_sd_mos(bra_mos, bra_exc[0])
    # MO overlaps
    mo_ovlp_mat = moovlp(sd_mos, sd_mos, S_AO)
    # Slater determinant overlap
    sd_ovlp = np.linalg.det(mo_ovlp_mat)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    run()
