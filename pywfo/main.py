#!/usr/bin/env python3

# http://dx.doi.org/10.1021/acs.jctc.5b01148

import io

import itertools as it
import numpy as np

from pysisyphus.calculators.WFOWrapper2 import WFOWrapper2


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


def perturb_mat(mat, scale=2e-1):
    """Add (small) random perturbations to the given matrix."""
    return mat + np.random.rand(*mat.shape)*scale


def run():
    np.set_printoptions(suppress=True, precision=6)

    # 4 MOs, 2 states
    dim_ = 4
    occ = 2
    ci_thresh = .5
    states = 2

    # Number of virtual orbitals and electrons (total, alpha, beta)
    virt = dim_ - occ
    nel = 2*occ
    nalpha = occ
    nbeta = nalpha

    # Indices of the occupied MOs
    mo_mask = np.arange(occ)

    def get_sd_mo_inds(exc=None):
        """Get MOs to form a Slater determinant for the given excitations."""
        if exc is None:
            return (mo_mask, ), (mo_mask, )

        # Assume excitation of beta electron
        all_beta_sd_mos = list()
        for exc_from, exc_to in zip(*exc):
            beta_inds = mo_mask.copy()
            beta_inds[exc_from] = exc_to + occ
            all_beta_sd_mos.append(beta_inds)
        return (mo_mask, ), all_beta_sd_mos

    moovlp = get_mo_ovlp((dim_, dim_))

    # Construct dummy MOs
    np.random.seed(20180325)
    _ = np.random.rand(dim_, dim_)
    bra_mos, _ = np.linalg.qr(_, mode="complete")
    # MOs are given per row
    bra_mos = bra_mos.T
    ket_mos, _ = np.linalg.qr(perturb_mat(bra_mos.T), mode="complete")
    ket_mos = ket_mos.T
    # ket_mos = bra_mos

    bra_mos_inv = np.linalg.inv(bra_mos)
    S_AO = bra_mos_inv.dot(bra_mos_inv.T)

    print("Bra MOs")
    print(bra_mos)
    print("Ket MOs")
    print(ket_mos)

    write_ref_data(bra_mos, ket_mos, S_AO)

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

    bra_cis, ket_cis = cis_

    # c = np.zeros((2,2))
    # c[1,0] = .7
    # c[1,1] = .6
    # ex_ = np.nonzero(c > ci_thresh)
    # _ = get_sd_mo_inds(bra_mos, exc=ex_)

    def get_sd_ovlps(bra_inds, ket_inds):
        ovlps = list()
        for bra_sd, ket_sd in it.product(bra_inds, ket_inds):
            b = bra_mos[bra_sd]
            k = ket_mos[ket_sd]
            ovlp_mat = moovlp(b, k, S_AO)
            ovlps.append(np.linalg.det(ovlp_mat))
        return ovlps

    ovlps = list()
    # Iterate over pairs of states and form the Slater determinants
    for bra_state, ket_state in it.product(bra_cis, ket_cis):
        bra_exc = np.nonzero(bra_state > ci_thresh)
        ket_exc = np.nonzero(ket_state > ci_thresh)

        # CI coefficients
        bra_coeffs = bra_state[bra_exc]
        ket_coeffs = ket_state[ket_exc]

        # MO indices of alpha and beta SDs
        bra_alpha, bra_beta = get_sd_mo_inds(bra_exc)
        ket_alpha, ket_beta = get_sd_mo_inds(ket_exc)

        # Slater determinant overlaps
        alpha_ovlps = get_sd_ovlps(bra_alpha, ket_alpha)
        beta_ovlps = get_sd_ovlps(bra_beta, ket_beta)

        alpha_ovlps = np.reshape(alpha_ovlps, (-1, ket_coeffs.size))
        beta_ovlps = np.reshape(beta_ovlps, (-1, ket_coeffs.size))

        # Contract with ket coefficients
        ket_contr = np.einsum("k,bk,bk->b", ket_coeffs, alpha_ovlps, beta_ovlps)
        braket_ovlp = (bra_coeffs * ket_contr).sum()
        ovlps.append(braket_ovlp)
    ovlps = np.array(ovlps)
    ovlps = ovlps.reshape(len(bra_cis), -1)
    print(ovlps)


if __name__ == "__main__":
    run()
