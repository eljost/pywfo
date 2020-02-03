#!/usr/bin/env python3

# http://dx.doi.org/10.1021/acs.jctc.5b01148

import itertools as it
import numpy as np


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
    np.set_printoptions(suppress=True, precision=4)

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

    def get_sd_mo_inds(exc=None):
        """Get MOs to form a Slater determinant for the given excitations."""
        if exc is None:
            return [mo_mask.copy(), ]

        all_sd_mos = list()
        for exc_from, exc_to in zip(*exc):
            mo_inds = mo_mask.copy()
            mo_inds[exc_from] = exc_to + occ
            all_sd_mos.append(mo_inds)
        return np.array(all_sd_mos)

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

    # c = np.zeros((2,2))
    # c[1,0] = .7
    # c[1,1] = .6
    # ex_ = np.nonzero(c > ci_thresh)
    # _ = get_sd_mo_inds(bra_mos, exc=ex_)

    # Iterate over pairs of states and form the Slater determinants
    for bra_state, ket_state in it.product(bra_cis, ket_cis):
        bra_exc = np.nonzero(bra_state > ci_thresh)
        ket_exc = np.nonzero(ket_state > ci_thresh)
        bra_mo_inds = get_sd_mo_inds(bra_exc)
        ket_mo_inds = get_sd_mo_inds(ket_exc)
        # Slater determinant overlaps
        for bra_sd, ket_sd in it.product(bra_mo_inds, ket_mo_inds):
            print("bra_sd", bra_sd)
            print("ket_sd", ket_sd)
            b = bra_mos[bra_sd]
            k = ket_mos[ket_sd]
            ovlp_mat = moovlp(b, k, S_AO)
            ovlp = np.linalg.det(ovlp_mat)
            print(ovlp)
            print()


if __name__ == "__main__":
    run()
