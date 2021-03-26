# [1] https://pubs.acs.org/doi/10.1021/acs.jctc.5b01148
#     Efficient and Flexible Computation of Many-Electron Wave Function Overlaps
#     Plasser, 2016


import itertools as it
import logging
from math import sqrt

import numba
import numpy as np


logger = logging.getLogger("pywfo")
log_funcs = {
    "info": logger.info,
}


def log(message, level="info"):
    log_funcs[level](message)


def moovlp(mos1, mos2, S_AO):
    """Overlap between two sets of MOs.

    <phi_p|phi_q> = sum_{u,v} C_{pu} C'_{qv} <X_u|X'_v>
    """

    # S_AO == None will only work if mos1 comprise all MOs (occ. + virt.).
    # If we only supply a subset of MOs like in an excited state SD, then this
    # will not work. So I deactivate it for now.
    # if S_AO is None:
    # mos1_inv = np.linalg.inv(mos1)
    # S_AO = mos1_inv.dot(mos1_inv.T)

    ovlp = np.einsum(
        "pu,qv,uv->pq", mos1, mos2, S_AO, optimize=["einsum_path", (0, 2), (0, 1)]
    )
    return ovlp


def moovlp_expl(mos1, mos2, S_AO):
    """Overlap between two sets of MOs.

    Explicit loops.

    <phi_p|phi_q> = sum_{u,v} C_{pu} C'_{qv} <X_u|X'_v>
    """

    P, _ = mos1.shape
    _, Q = mos2.shape
    U, V = S_AO.shape

    pv = np.zeros((P, V))
    for p in range(P):
        for v in range(V):
            pv[p, v] = (S_AO[:, v] * mos1[p, :]).sum()

    pq = np.zeros((P, Q))
    for p in range(P):
        for q in range(Q):
            pq[p, q] = (pv[p, :] * mos2[q, :]).sum()

    return pq


@numba.jit(nopython=True)
def moovlp_dots(mos1, mos2, S_AO):
    """Overlap between two sets of MOs.

    <phi_p|phi_q> = sum_{u,v} C_{pu} C'_{qv} <X_u|X'_v>
    """

    return mos1.dot(S_AO).dot(mos2.T)


def overlaps(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    with_gs=True,
    ao_ovlps="ket",
):
    assert bra_mos.shape == ket_mos.shape
    assert bra_ci.shape == ket_ci.shape

    # Indices of the occupied MOs that make up the ground state Slater
    # determinant.
    occ_mos = np.arange(occ)

    if isinstance(ao_ovlps, str):
        inv = np.linalg.inv({"bra": bra_mos, "ket": ket_mos}[ao_ovlps])
        S_AO = inv.dot(inv.T)
    elif isinstance(ao_ovlps, np.array):
        S_AO = ao_ovlps
    else:
        raise Exception("Invalid AO overlaps!")

    # Collect possible overlaps
    unique_ovlps = set()
    per_state = list()

    def count_slater_dets(ci_coeffs, infix):
        # Select CI coefficients above the given threshold
        # and drop first dimension (number of states)
        _, *ci_above_thresh = np.nonzero(np.abs(ci_coeffs) >= ci_thresh)
        sds = set(tuple(zip(*ci_above_thresh)))
        log(f"{len(sds)} {infix} SDs")
        return sds

    # Determine unique bra and ket-SDs
    bra_sds = count_slater_dets(bra_ci, "bra")
    ket_sds = count_slater_dets(ket_ci, "bra")

    for bra_state, ket_state in it.product(bra_ci, ket_ci):
        bra_exc = [(f, t) for f, t in zip(*np.nonzero(np.abs(bra_state) > ci_thresh))]
        ket_exc = [(f, t) for f, t in zip(*np.nonzero(np.abs(ket_state) > ci_thresh))]
        combinations = tuple(it.product(bra_exc, ket_exc))
        per_state.append(combinations)
        unique_ovlps |= set(combinations)

    # Construct GS-GS overlap and overlap between bra-SDs and ket-GS
    unique_ovlps |= {(None, None)}
    bra_ovlps = set([(bra_sd, None) for bra_sd in bra_sds])
    # as well as ket-SDS and bra-GS
    ket_ovlps = set([(None, ket_sd) for ket_sd in ket_sds])
    unique_ovlps |= bra_ovlps
    unique_ovlps |= ket_ovlps

    log(f"bra SDs: {len(bra_sds)}")
    log(f"ket SDs: {len(ket_sds)}")
    log(f"SD pairs: {len(bra_sds)*len(ket_sds)}")
    log(f"Unique overlaps: {len(unique_ovlps)}")

    # Form the SDs by gathering appropriate MOs and determine their sign
    # after sorting. The SDs are sorted in a way, that the accepting orbital
    # of the excitation (hole orbital) is the last one.
    occ_mo_list = list(occ_mos)
    slater_dets = dict()

    def form_sd(inds, braket):
        from_, to = inds
        mo_inds = occ_mo_list.copy()
        # Delete particle orbital
        mo_inds.remove(from_)
        # Add hole orbital
        mo_inds.append(to + occ)
        sign = (-1) ** (occ - from_ + 1)
        slater_dets[(braket, inds)] = (sign, mo_inds)

    [form_sd(bra_sd, "bra") for bra_sd in bra_sds]
    [form_sd(ket_sd, "ket") for ket_sd in ket_sds]
    # Add ground state SDs without any excitations
    slater_dets[("bra", None)] = (1, occ_mo_list.copy())
    slater_dets[("ket", None)] = (1, occ_mo_list.copy())

    # TODO: precompute minors

    # Precontract bra_mos, S_AO and ket_mos
    mo_ovlps = bra_mos.dot(S_AO).dot(ket_mos.T)
    mo_ovlp_det = np.linalg.det(mo_ovlps)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    sd_ovlps = dict()
    # Calculate overlaps between SDs
    for (bra, ket) in unique_ovlps:
        bra_sign, bra_inds = slater_dets[("bra", bra)]
        ket_sign, ket_inds = slater_dets[("ket", ket)]
        ovlp_mat = mo_ovlps[bra_inds][:, ket_inds]
        ovlp_mat *= bra_sign * ket_sign
        sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)

    for k, v in sd_ovlps.items():
        print(k, v)

    state_inds = it.product(range(bra_ci.shape[0]), range(ket_ci.shape[0]))
    wf_ovlps = list()
    none_none = sd_ovlps[(None), (None)]
    for (i, j), a in zip(state_inds, per_state):
        bra_inds, ket_inds = zip(*a)
        bra_coeffs = np.array([bra_ci[i][bi] for bi in bra_inds])
        ket_coeffs = np.array([ket_ci[j][ki] for ki in ket_inds])
        bra_none = np.array([sd_ovlps[(bi, None)] for bi in bra_inds])
        none_ket = np.array([sd_ovlps[(None, ki)] for ki in ket_inds])
        bra_ket = np.array([sd_ovlps[_] for _ in a])
        wf_ovlps.append(
            (
                bra_coeffs * ket_coeffs * [none_none * bra_ket - bra_none * none_ket]
            ).sum()
        )
    wf_ovlps = np.reshape(wf_ovlps, (len(bra_ci), -1))

    return wf_ovlps


def overlaps_naive(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    with_gs=True,
    ao_ovlps="ket",
):
    if isinstance(ao_ovlps, str):
        inv = np.linalg.inv({"bra": bra_mos, "ket": ket_mos}[ao_ovlps])
        S_AO = inv.dot(inv.T)
    elif isinstance(ao_ovlps, np.array):
        S_AO = ao_ovlps
    else:
        raise Exception("Invalid AO overlaps!")

    # MO overlaps
    mo_ovlps = bra_mos.dot(S_AO).dot(ket_mos.T)
    mo_ovlp_det = np.linalg.det(mo_ovlps)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    # ovlp_mat = mo_ovlps[bra_inds][:, ket_inds]
    # sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(mo_ovlps[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    def form_sd(inds):
        from_, to = inds
        mo_inds_ = mo_inds.copy()
        # Delete particle orbital
        mo_inds_.remove(from_)
        # Add hole orbital
        mo_inds_.append(to + occ)
        sign = (-1) ** (occ - from_ + 1)
        # return sign, mo_inds_
        return sign, mo_inds_

    prefac = 1 / sqrt(2)

    def sds_from_restricted_tden(inds):
        sign, sd = form_sd(inds)
        # Return plus and minus SD
        p_prefac = prefac
        m_prefac = -prefac
        return (p_prefac, sd), (m_prefac, sd)

    # ket_prefac_sds = it.chain(*)

    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        ci_i_above_thresh = above_thresh(ci_i)
        # Over all ket states
        for j, ci_j in enumerate(ket_ci):
            print(f"overlap between bra {i}, and ket {j}")
            ci_j_above_thresh = above_thresh(ci_j)
            # Over all SDs in the bra state
            bra_prefac_sds = list(
                it.chain(
                    *[
                        sds_from_restricted_tden(sd_k)
                        for sd_k in zip(*ci_i_above_thresh)
                    ]
                )
            )
            for k, sd_k in enumerate(zip(*ci_i_above_thresh)):
                k_from, k_to = sd_k
                ci_k_coeff = ci_i[k_from, k_to]
                # Over all SDs in the ket state
                for l, sd_l in enumerate(zip(*ci_j_above_thresh)):
                    l_from, l_to = sd_l
                    ci_l_coeff = ci_j[l_from, l_to]
                    print(f"sd_k: {sd_k}, {ci_k_coeff:.6f}")
                    print(f"sd_l: {sd_l}, {ci_l_coeff:.6f}")
                    # SD overlaps
                    # alpha overlaps
                    k_sign, k_alpha_inds = form_sd(sd_k)
                    print(f"\t{k_alpha_inds}, sign {k_sign}")
                    l_sign, l_alpha_inds = form_sd(sd_l)
                    print(f"\t{l_alpha_inds}, sign {l_sign}")
                    alpha_ovlp_mat = mo_ovlps[k_alpha_inds][:, l_alpha_inds]
                    alpha_ovlp = k_sign * l_sign * np.linalg.det(alpha_ovlp_mat)
                    # beta overlaps. For closed shell molecules the beta overlap
                    # will always be the same, that is the overlap of the GS
                    # configuration.
                    ovlps[i, j] += ci_k_coeff * ci_l_coeff * alpha_ovlp * beta_ovlp
                print()
    return ovlps


def overlaps_naive(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    with_gs=True,
    ao_ovlps="ket",
):
    if isinstance(ao_ovlps, str):
        inv = np.linalg.inv({"bra": bra_mos, "ket": ket_mos}[ao_ovlps])
        S_AO = inv.dot(inv.T)
    elif isinstance(ao_ovlps, np.array):
        S_AO = ao_ovlps
    else:
        raise Exception("Invalid AO overlaps!")

    # MO overlaps
    mo_ovlps = bra_mos.dot(S_AO).dot(ket_mos.T)
    mo_ovlp_det = np.linalg.det(mo_ovlps)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    # ovlp_mat = mo_ovlps[bra_inds][:, ket_inds]
    # sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(mo_ovlps[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    def form_sd(inds):
        from_, to = inds
        mo_inds_ = mo_inds.copy()
        # Delete particle orbital
        mo_inds_.remove(from_)
        # Add hole orbital
        mo_inds_.append(to + occ)
        sign = (-1) ** (occ - from_ + 1)
        # return sign, mo_inds_
        return sign, mo_inds_

    prefac = 1 / sqrt(2)
    def sds_from_restricted_tden(inds, ci_coeff):
        sign, sd = form_sd(inds)
        # Return plus and minus SD
        p_prefac = prefac * ci_coeff  # alpha excitation
        m_prefac = -prefac * ci_coeff  # beta excitation
        return (p_prefac, sd, mo_inds), (m_prefac, mo_inds, sd)

    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        ci_i_above_thresh = above_thresh(ci_i)
        ci_coeffs_i = ci_i[ci_i_above_thresh]
        bra_prefac_sds = list(
            it.chain(
                *[
                    sds_from_restricted_tden(sd_k, ci_coeff)
                    for ci_coeff, *sd_k in zip(ci_coeffs_i, *ci_i_above_thresh)
                ]
            )
        )
        # Over all ket states
        for j, ci_j in enumerate(ket_ci):
            print(f"overlap between bra {i}, and ket {j}")
            ci_j_above_thresh = above_thresh(ci_j)
            ci_coeffs_j = ci_j[ci_j_above_thresh]
            ket_prefac_sds = list(
                it.chain(
                    *[
                        sds_from_restricted_tden(sd_l, ci_coeff)
                        for ci_coeff, *sd_l in zip(ci_coeffs_j, *ci_j_above_thresh)
                    ]
                )
            )
            # Over all SDs in the bra state
            for prefac_k, sd_k_a, sd_k_b in bra_prefac_sds:
                # Over all SDs in the ket state
                for prefac_l, sd_l_a, sd_l_b in ket_prefac_sds:
                    print(f"sd_k: {sd_k_a}, {sd_k_b}, {prefac_k:.6f}")
                    print(f"sd_l: {sd_l_a}, {sd_l_b}, {prefac_l:.6f}")
                    alpha_ovlp_mat = mo_ovlps[sd_k_a][:, sd_l_a]
                    beta_ovlp_mat = mo_ovlps[sd_k_b][:, sd_l_b]
                    alpha_ovlp = prefac_k * prefac_l * np.linalg.det(alpha_ovlp_mat)
                    beta_ovlp = prefac_k * prefac_l * np.linalg.det(beta_ovlp_mat)
                    ovlps[i, j] += 2*alpha_ovlp * beta_ovlp
                print()
    return ovlps
