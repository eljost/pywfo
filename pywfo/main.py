# [1] https://pubs.acs.org/doi/10.1021/acs.jctc.5b01148
#     Efficient and Flexible Computation of Many-Electron Wave Function Overlaps
#     Plasser, 2016


import functools
import itertools as it
import logging
from math import sqrt

import numba
import numpy as np


np.set_printoptions(precision=4, suppress=True)

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
    S_MO = bra_mos.dot(S_AO).dot(ket_mos.T)
    mo_ovlp_det = np.linalg.det(S_MO)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    sd_ovlps = dict()
    # Calculate overlaps between SDs
    for (bra, ket) in unique_ovlps:
        bra_sign, bra_inds = slater_dets[("bra", bra)]
        ket_sign, ket_inds = slater_dets[("ket", ket)]
        ovlp_mat = S_MO[bra_inds][:, ket_inds]
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
    S_MO = bra_mos.dot(S_AO).dot(ket_mos.T)
    mo_ovlp_det = np.linalg.det(S_MO)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    # ovlp_mat = S_MO[bra_inds][:, ket_inds]
    # sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(S_MO[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    def form_exc_inds(inds):
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
        sign, exc_inds = form_exc_inds(inds)
        # Return plus and minus SD
        return (prefac, exc_inds, mo_inds), (-prefac, mo_inds, exc_inds)

    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        ci_i_above_thresh = above_thresh(ci_i)
        ci_coeffs_i = ci_i[ci_i_above_thresh]
        bra_prefac_sds = list(
            it.chain(
                *[sds_from_restricted_tden(sd_k) for sd_k in zip(*ci_i_above_thresh)]
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
                        sds_from_restricted_tden(sd_l)
                        for sd_l in zip(*ci_j_above_thresh)
                    ]
                )
            )
            # Over all SDs in bra state
            ij_increments = list()
            pfk = list()
            pfl = list()
            for prefac_k, sd_k_a, sd_k_b in bra_prefac_sds:
                pfk.append(prefac_k)
                print(f"sd_k: {sd_k_a}, {sd_k_b}, {prefac_k:.6f}")
                ij_increment = 0.0
                # Over all SDs in ket state
                for prefac_l, sd_l_a, sd_l_b in ket_prefac_sds:
                    pfl.append(prefac_k)
                    print(f"sd_l: {sd_l_a}, {sd_l_b}, {prefac_l:.6f}")
                    alpha_ovlp_mat = S_MO[sd_k_a][:, sd_l_a]
                    beta_ovlp_mat = S_MO[sd_k_b][:, sd_l_b]
                    alpha_ovlp = prefac_l * np.linalg.det(alpha_ovlp_mat)
                    beta_ovlp = prefac_l * np.linalg.det(beta_ovlp_mat)
                    ij_increment += alpha_ovlp * beta_ovlp
                ij_increments.append(prefac_k ** 2 * ij_increment)
            ij_increments = np.array(ij_increments)
            ovlps[i, j] = 2 * (ci_coeffs_i * ci_coeffs_j * ij_increments).sum()
            print()
    return ovlps


def get_S_AO(bra_mos, ket_mos, ao_ovlps):
    if isinstance(ao_ovlps, str):
        inv = np.linalg.inv({"bra": bra_mos, "ket": ket_mos}[ao_ovlps])
        S_AO = inv.dot(inv.T)
    elif isinstance(ao_ovlps, np.array):
        S_AO = ao_ovlps
    else:
        raise Exception("Invalid AO overlaps!")
    print("S_AO")
    print(S_AO)
    return S_AO


def overlaps_naive(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    ao_ovlps="ket",
):
    S_AO = get_S_AO(bra_mos, ket_mos, ao_ovlps)

    # MO overlaps
    S_MO = bra_mos.dot(S_AO).dot(ket_mos.T)
    moo_str = np.array2string(S_MO, precision=4)
    print("MO Overlaps")
    print(moo_str)
    mo_ovlp_det = np.linalg.det(S_MO)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    # ovlp_mat = S_MO[bra_inds][:, ket_inds]
    # sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(S_MO[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    def form_exc_inds(inds):
        from_, to = inds
        exc_inds = mo_inds.copy()
        # Delete particle orbital
        exc_inds.remove(from_)
        # Add hole orbital
        exc_inds.append(to + occ)
        # Exchanging two electrons flips the sign of the determinant
        return exc_inds

    prefac = 1 / sqrt(2)

    def sds_from_restricted_tden(inds, ci_coeff):
        exc_inds = form_exc_inds(inds)
        # Return plus and minus SD
        prefac_ = prefac * ci_coeff
        return (prefac_, exc_inds, mo_inds), (prefac_, mo_inds, exc_inds)

    log(f"Bra states: {bra_ci.shape[0]}")
    log(f"Ket states: {ket_ci.shape[0]}")

    fmt = " .6f"
    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        i_above = above_thresh(ci_i)
        bra_prefac_sds = list(
            it.chain(
                *[
                    sds_from_restricted_tden(sd_k, ci_coeff)
                    for *sd_k, ci_coeff in zip(*i_above, ci_i[i_above])
                ]
            )
        )
        # Over all ket states
        for j, ci_j in enumerate(ket_ci):
            print(f"overlap between bra {i}, and ket {j}")
            j_above = above_thresh(ci_j)
            # Don't use iterator here, as we must iterate multiple times
            # over this list
            ket_prefac_sds = list(
                it.chain(
                    *[
                        sds_from_restricted_tden(sd_l, ci_coeff)
                        for *sd_l, ci_coeff in zip(*j_above, ci_j[j_above])
                    ]
                )
            )
            # Over all SDs in bra state
            ij_increments = list()
            pfk = list()
            for prefac_k, sd_k_a, sd_k_b in bra_prefac_sds:
                pfk.append(prefac_k)
                print(f"sd_k: {sd_k_a}, {sd_k_b}, {prefac_k:.6f}")
                ij_increment = 0.0
                # Over all SDs in ket state
                for prefac_l, sd_l_a, sd_l_b in ket_prefac_sds:
                    print(f"sd_l: {sd_l_a}, {sd_l_b}, {prefac_l:.6f}")
                    alpha_ovlp_mat = S_MO[sd_k_a][:, sd_l_a]
                    beta_ovlp_mat = S_MO[sd_k_b][:, sd_l_b]
                    alpha_ovlp = np.linalg.det(alpha_ovlp_mat)
                    beta_ovlp = np.linalg.det(beta_ovlp_mat)
                    print(
                        f"\tpfk={prefac_k:{fmt}}, pfl={prefac_l:{fmt}}, "
                        f"|alpha|={alpha_ovlp:{fmt}}, |beta|={beta_ovlp:{fmt}}"
                    )
                    ij_increment += prefac_l * alpha_ovlp * beta_ovlp
                    # print("det", prefac_l * alpha_ovlp * beta_ovlp)
                    print("det", alpha_ovlp * beta_ovlp)
                ij_increments.append(ij_increment)
            ij_increments = np.array(ij_increments)
            pfk = np.array(pfk)
            ovlps[i, j] = (pfk * ij_increments).sum()
            print()
    return ovlps


def overlaps_most_naive(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    ao_ovlps="ket",
):
    S_AO = get_S_AO(bra_mos, ket_mos, ao_ovlps)

    # MO overlaps
    S_MO = bra_mos.dot(S_AO).dot(ket_mos.T)
    moo_str = np.array2string(S_MO, precision=4)
    print("MO Overlaps")
    print(moo_str)
    mo_ovlp_det = np.linalg.det(S_MO)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(S_MO[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    def form_exc_inds(inds):
        from_, to = inds
        exc_inds = mo_inds.copy()
        # Delete particle orbital
        exc_inds.remove(from_)
        # Add hole orbital
        exc_inds.append(to + occ)
        return exc_inds

    prefac = 1 / sqrt(2)

    def sds_from_restricted_tden(inds, ci_coeff):
        exc_inds = form_exc_inds(inds)
        prefac_ = prefac * ci_coeff
        # Return plus and minus SD
        return (prefac_, exc_inds, mo_inds), (prefac_, mo_inds, exc_inds)

    log(f"Bra states: {bra_ci.shape[0]}")
    log(f"Ket states: {ket_ci.shape[0]}")

    fmt = " .6f"
    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        i_above = above_thresh(ci_i)
        bra_prefac_sds = list(
            it.chain(
                *[
                    sds_from_restricted_tden(sd_k, ci_coeff)
                    for *sd_k, ci_coeff in zip(*i_above, ci_i[i_above])
                ]
            )
        )
        # Over all ket states
        for j, ci_j in enumerate(ket_ci):
            print(f"overlap between bra {i}, and ket {j}")
            j_above = above_thresh(ci_j)
            # Don't use iterator here, as we must iterate multiple times
            # over this list
            ket_prefac_sds = list(
                it.chain(
                    *[
                        sds_from_restricted_tden(sd_l, ci_coeff)
                        for *sd_l, ci_coeff in zip(*j_above, ci_j[j_above])
                    ]
                )
            )
            ij = 0
            for prefac_k, sd_k_a, sd_k_b in bra_prefac_sds:
                print(f"sd_k: {sd_k_a}, {sd_k_b}, {prefac_k:.6f}")
                # Over all SDs in ket state
                for prefac_l, sd_l_a, sd_l_b in ket_prefac_sds:
                    print(f"\tsd_l: {sd_l_a}, {sd_l_b}, {prefac_l:.6f}")
                    alpha_ovlp_mat = S_MO[sd_k_a][:, sd_l_a]
                    beta_ovlp_mat = S_MO[sd_k_b][:, sd_l_b]
                    alpha_ovlp = np.linalg.det(alpha_ovlp_mat)
                    beta_ovlp = np.linalg.det(beta_ovlp_mat)
                    print(
                        f"\tpfk={prefac_k:{fmt}}, pfl={prefac_l:{fmt}}, "
                        f"|alpha|={alpha_ovlp:{fmt}}, |beta|={beta_ovlp:{fmt}}"
                    )
                    ij += prefac_k * prefac_l * alpha_ovlp * beta_ovlp
                    print("\tij", ij)
            print("ij", ij)
            ovlps[i, j] = ij
            print()
    return ovlps


def overlaps_cache(
    bra_mos,
    ket_mos,
    bra_ci,
    ket_ci,
    occ,
    ci_thresh=1e-4,
    ao_ovlps="ket",
):
    S_AO = get_S_AO(bra_mos, ket_mos, ao_ovlps)

    # MO overlaps
    S_MO = bra_mos.dot(S_AO).dot(ket_mos.T)
    moo_str = np.array2string(S_MO, precision=4)
    print("MO Overlaps")
    print(moo_str)
    mo_ovlp_det = np.linalg.det(S_MO)
    print(f"Determinant of MO-overlap matrix: {mo_ovlp_det:.10f}")

    def above_thresh(ci_coeffs):
        return np.nonzero(np.abs(ci_coeffs) >= ci_thresh)

    ovlps = np.zeros((bra_ci.shape[0], ket_ci.shape[0]))

    # Overlap of the GS configurations
    # ovlp_mat = S_MO[bra_inds][:, ket_inds]
    # sd_ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    mo_inds = list(range(occ))
    gs_ovlp = np.linalg.det(S_MO[mo_inds][:, mo_inds])
    beta_ovlp = gs_ovlp
    print(f"GS ovlp: {gs_ovlp*gs_ovlp:.6f}")

    prefac = 1 / sqrt(2)

    def sds_from_restricted_tden(inds, ci_coeff):
        prefac_ = prefac * ci_coeff
        # Return +alpha excitation and -beta excitation. The -beta SD
        # is resorted once and again changes its sign.
        tpl = tuple(inds)
        print(prefac_)
        return (prefac_, tpl, None), (prefac_, None, tpl)

    log(f"Bra states: {bra_ci.shape[0]}")
    log(f"Ket states: {ket_ci.shape[0]}")

    def form_exc_inds(inds):
        try:
            from_, to = inds
        except TypeError:
            return mo_inds
        exc_inds = mo_inds.copy()
        # Delete particle orbital
        exc_inds.remove(from_)
        # Add hole orbital
        exc_inds.append(to + occ)
        # Exchanging two electrons flips the sign of the determinant
        return exc_inds


    @functools.cache
    def sd_ovlp(sd_k, sd_l):
        mo_inds_k = form_exc_inds(sd_k)
        mo_inds_l = form_exc_inds(sd_l)
        mo_ovlp_mat = S_MO[mo_inds_k][:, mo_inds_l]
        det = np.linalg.det(mo_ovlp_mat)
        return det

    fmt = " .6f"
    # Over all bra states
    for i, ci_i in enumerate(bra_ci):
        i_above = above_thresh(ci_i)
        bra_prefac_sds = list(
            it.chain(
                *[
                    sds_from_restricted_tden(sd_k, ci_coeff)
                    for *sd_k, ci_coeff in zip(*i_above, ci_i[i_above])
                ]
            )
        )
        # Over all ket states
        for j, ci_j in enumerate(ket_ci):
            j_above = above_thresh(ci_j)
            # Don't use iterator here, as we must iterate multiple times
            # over this list
            ket_prefac_sds = list(
                it.chain(
                    *[
                        sds_from_restricted_tden(sd_l, ci_coeff)
                        for *sd_l, ci_coeff in zip(*j_above, ci_j[j_above])
                    ]
                )
            )
            # Over all SDs in bra state
            ij_increments = list()
            pfk = list()
            for prefac_k, sd_k_a, sd_k_b in bra_prefac_sds:
                pfk.append(prefac_k)
                ij_increment = 0.0
                # Over all SDs in ket state
                for prefac_l, sd_l_a, sd_l_b in ket_prefac_sds:
                    alpha_ovlp = sd_ovlp(sd_k_a, sd_l_a)
                    beta_ovlp = sd_ovlp(sd_k_b, sd_l_b)
                    ij_increment += prefac_l * alpha_ovlp * beta_ovlp
                ij_increments.append(ij_increment)
            ij_increments = np.array(ij_increments)
            pfk = np.array(pfk)
            ovlps[i, j] = (pfk * ij_increments).sum()
    return ovlps
