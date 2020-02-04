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


def overlaps(bra_mos, ket_mos, bra_ci, ket_ci, occ, ci_thresh=.5):
    # 4 MOs, 2 states
    assert bra_mos.shape == ket_mos.shape
    assert bra_ci.shape == ket_ci.shape

    dim_ = bra_mos.shape[0]
    states = bra_ci.shape[0]

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

        alpha_mos = list()
        beta_mos = list()
        exc_sd_mo_inds = list()
        for exc_from, exc_to in zip(*exc):
            mo_inds = mo_mask.copy()
            mo_inds[exc_from] = exc_to + occ
            alpha_mos.extend([mo_mask, mo_inds])
            beta_mos.extend([mo_inds, mo_mask])
        return alpha_mos, beta_mos

    moovlp = get_mo_ovlp((dim_, dim_))

    bra_mos_inv = np.linalg.inv(bra_mos)
    S_AO = bra_mos_inv.dot(bra_mos_inv.T)

    print("Bra MOs")
    print(bra_mos)
    print("Ket MOs")
    print(ket_mos)

    write_ref_data(bra_mos, ket_mos, S_AO)

    def get_sd_ovlps(bra_inds, ket_inds):
        ovlps = list()
        for bra_sd, ket_sd in it.product(bra_inds, ket_inds):
            b = bra_mos[bra_sd]
            k = ket_mos[ket_sd]
            ovlp_mat = moovlp(b, k, S_AO)
            ovlps.append(np.linalg.det(ovlp_mat))
        return ovlps

    ovlps = list()
    _ = 1/(2**0.5)
    spin_adapt = np.array((_, -_))[None,:]
    # Iterate over pairs of states
    for bra_state, ket_state in it.product(bra_ci, ket_ci):
        # Select CI coefficients above the given threshold
        bra_exc = np.nonzero(bra_state > ci_thresh)
        ket_exc = np.nonzero(ket_state > ci_thresh)

        # CI coefficients
        bra_coeffs = bra_state[bra_exc]
        ket_coeffs = ket_state[ket_exc]
        # Spin adapt the CI coefficients.
        # Every coefficient yields two spin adapted coefficients
        # weighted by 1/sqrt(2).
        bra_coeffs = (bra_coeffs[:,None] * spin_adapt).flatten()
        ket_coeffs = (ket_coeffs[:,None] * spin_adapt).flatten()

        # Get the MO indices that make up the (excited) Slater determinants,
        # separated by spin.
        bra_alpha, bra_beta = get_sd_mo_inds(bra_exc)
        ket_alpha, ket_beta = get_sd_mo_inds(ket_exc)

        # Calculate the overlap between the SDs for alpha and beta
        # orbitals separately.
        alpha_ovlps = get_sd_ovlps(bra_alpha, ket_alpha)
        beta_ovlps = get_sd_ovlps(bra_beta, ket_beta)

        alpha_ovlps = np.reshape(alpha_ovlps, (-1, ket_coeffs.size))
        beta_ovlps = np.reshape(beta_ovlps, (-1, ket_coeffs.size))

        # Contract with ket coefficients
        ket_contr = np.einsum("k,bk,bk->b", ket_coeffs, alpha_ovlps, beta_ovlps)
        braket_ovlp = (bra_coeffs * ket_contr).sum()
        # _, __ = np.einsum_path("b,k,bk,bk", bra_coeffs, ket_coeffs, alpha_ovlps, beta_ovlps)
        # import pdb; pdb.set_trace()
        # braket_ovlp = np.einsum("b,k,bk,bk", bra_coeffs, ket_coeffs, alpha_ovlps, beta_ovlps)
        # print(_)
        ovlps.append(braket_ovlp)
    ovlps = np.array(ovlps)
    ovlps = ovlps.reshape(len(bra_ci), -1)
    print(ovlps)

    return ovlps
