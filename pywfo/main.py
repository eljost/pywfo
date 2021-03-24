# [1] https://pubs.acs.org/doi/10.1021/acs.jctc.5b01148
#     Efficient and Flexible Computation of Many-Electron Wave Function Overlaps
#     Plasser, 2016


import itertools as it

import numba
import numpy as np


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
    bra_mos, ket_mos, bra_ci, ket_ci, occ, ci_thresh=0.5, with_gs=False, ao_ovlps="bra"
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
    for bra_state, ket_state in it.product(bra_ci, ket_ci):
        # Select CI coefficients above the given threshold
        bra_exc = [(f, t) for f, t in zip(*np.nonzero(np.abs(bra_state) > ci_thresh))]
        ket_exc = [(f, t) for f, t in zip(*np.nonzero(np.abs(ket_state) > ci_thresh))]
        combinations = tuple(it.product(bra_exc, ket_exc))
        per_state.append(combinations)
        unique_ovlps |= set(combinations)

    # Determine unique bra and ket-SDs
    bra_sds, ket_sds = zip(*unique_ovlps)
    bra_sds = set(bra_sds)
    ket_sds = set(ket_sds)

    bra_ovlps = set([(bra_sd, None) for bra_sd in bra_sds])
    ket_ovlps = set([(None, ket_sd) for ket_sd in ket_sds])
    unique_ovlps |= bra_ovlps
    unique_ovlps |= ket_ovlps
    unique_ovlps |= {(None, None)}

    print(f"bra SDs: {len(bra_sds)}")
    print(f"ket SDs: {len(ket_sds)}")
    print(f"SD pairs: {len(bra_sds)*len(ket_sds)}")
    print(f"Unique overlaps: {len(unique_ovlps)}")

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

    ovlps = dict()
    for (bra, ket) in unique_ovlps:
        bra_sign, bra_inds = slater_dets[("bra", bra)]
        ket_sign, ket_inds = slater_dets[("ket", ket)]
        ovlp_mat = mo_ovlps[bra_inds][:, ket_inds]
        ovlp_mat *= bra_sign * ket_sign
        ovlps[(bra, ket)] = np.linalg.det(ovlp_mat)
    # for k,v in ovlps.items(): print(k, f"{v: >10.6f}")

    state_inds = it.product(range(bra_ci.shape[0]), range(ket_ci.shape[0]))
    wf_ovlps = list()
    none_none = ovlps[(None), (None)]
    for (i, j), a in zip(state_inds, per_state):
        bra_inds, ket_inds = zip(*a)
        bra_coeffs = np.array([bra_ci[i][bi] for bi in bra_inds])
        ket_coeffs = np.array([ket_ci[j][ki] for ki in ket_inds])
        bra_none = np.array([ovlps[(bi, None)] for bi in bra_inds])
        none_ket = np.array([ovlps[(None, ki)] for ki in ket_inds])
        bra_ket = np.array([ovlps[_] for _ in a])
        wf_ovlps.append(
            (
                bra_coeffs * ket_coeffs * [none_none * bra_ket - bra_none * none_ket]
            ).sum()
        )
    wf_ovlps = np.reshape(wf_ovlps, (len(bra_ci), -1))

    return wf_ovlps
