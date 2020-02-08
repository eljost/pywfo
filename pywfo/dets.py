from collections import namedtuple

import numpy as np


np.set_printoptions(suppress=True, precision=8, linewidth=120)


BlockResult = namedtuple(
    # "BlockResult", "block_num super_block_num blocks block_map super_map " \
                   # "sort_inds ci_coeffs"
    "BlockResult", "block_num super_block_num blocks block_map super_map " \
                   "sort_inds"
)

def block_dets(dets, ci_coeffs):
    # Prefix with indices so we get the actual sorting
    ind_dets = [(i, d) for i, d in enumerate(dets)]
    sort_inds, dets = zip(*sorted(ind_dets, key=lambda x: x[1]))
    sort_inds = list(sort_inds)

    # # Blocks
    # blocks = dict.fromkeys(dets)
    # # Superblocks
    # super_blocks = dict.fromkeys([_[:-1] for _ in dets])

    block_ind = 0
    super_block_ind = 0
    blocks = [list(dets[0]), ]
    block_map = [block_ind, ]
    super_map = [super_block_ind, ]
    for i, sd in enumerate(dets[1:]):
        if not sd == dets[i]:
            block_ind += 1
            blocks.append(list(sd))
            if not sd[:-1] == dets[i][:-1]:
                super_block_ind += 1
                super_map.append(block_ind)
        block_map.append(block_ind)

    res = BlockResult(
            block_ind+1,
            super_block_ind+1,
            blocks,
            block_map,
            super_map,
            sort_inds,
    )
    return res


def dbg_print(blocked, coeffs):
    sis = blocked.sort_inds
    blocks = blocked.blocks
    block_map = blocked.block_map
    for i, block in enumerate(blocked.block_map):
        si = sis[i]
        blk = block_map[i]
        print(i, si, blk, coeffs[si,0], blocks[block][15:])
    print()


def precompute(bra_blocked, ket_blocked, mo_ovlps):
    block_ovlps = np.zeros((bra_blocked.block_num, ket_blocked.block_num))

    for i, bra_block in enumerate(bra_blocked.blocks):
        for j, ket_block in enumerate(ket_blocked.blocks):
            ovlp_mat = mo_ovlps[bra_block][:,ket_block]
            block_ovlps[i,j] = np.linalg.det(ovlp_mat)
    return block_ovlps


def wfoverlap(bra_mos, ket_mos, bra_ci, ket_ci, ci_thresh, S_AO):
    bra_states = bra_ci.shape[0]
    ket_states = ket_ci.shape[0]

    bra_dets, bra_inds, bra_coeffs = get_dets(bra_ci, ci_thresh)
    ket_dets, ket_inds, ket_coeffs = get_dets(ket_ci, ci_thresh)

    bra_det_str = get_dets_str(bra_dets, bra_coeffs)
    ket_det_str = get_dets_str(ket_dets, ket_coeffs)
    with open("bra_dets", "w") as handle:
        handle.write(bra_det_str)
    with open("ket_dets", "w") as handle:
        handle.write(ket_det_str)

    # Bra
    bra_alpha, bra_beta = zip(*[expand_det(det) for det in bra_dets])
    bra_alpha_blocked = block_dets(bra_alpha, bra_coeffs)
    bra_beta_blocked = block_dets(bra_beta, bra_coeffs)

    # Ket
    ket_alpha, ket_beta = zip(*[expand_det(det) for det in ket_dets])
    ket_alpha_blocked = block_dets(ket_alpha, ket_coeffs)
    ket_beta_blocked = block_dets(ket_beta, ket_coeffs)

    ba_blocks = bra_alpha_blocked.block_num
    ka_blocks = ket_alpha_blocked.block_num
    bb_blocks = bra_beta_blocked.block_num
    kb_blocks = ket_beta_blocked.block_num
    block_det_num = ba_blocks * ka_blocks + bb_blocks * kb_blocks
    print(f"<bra| alpha blocks: {ba_blocks: >16d}")
    print(f"|ket> alpha blocks: {ka_blocks: >16d}")
    print(f"<bra|  beta blocks: {bb_blocks: >16d}")
    print(f"|ket>  beta blocks: {kb_blocks: >16d}")
    print(f"Block determinants: {block_det_num: >16d}")
    print()

    # print(f"<bra| alpha super-blocks: {bra_alpha_blocked.super_block_num: >16d}")
    # print(f"|ket> alpha super-blocks: {ket_alpha_blocked.super_block_num: >16d}")
    # print(f"<bra|  beta super-blocks: {bra_beta_blocked.super_block_num: >16d}")
    # print(f"|ket>  beta super-blocks: {ket_beta_blocked.super_block_num: >16d}")

    # print("bra alpha")
    # dbg_print(bra_alpha_blocked, bra_coeffs)
    # print("bra beta")
    # dbg_print(bra_beta_blocked, bra_coeffs)
    # print("ket alpha")
    # dbg_print(ket_alpha_blocked, ket_coeffs)
    # print("ket beta")
    # dbg_print(ket_beta_blocked, ket_coeffs)

    def ci_block_map(block_map, sort_inds):
        _ = np.zeros_like(block_map, dtype=int)
        _[sort_inds] = block_map
        return _

    bac = ci_block_map(bra_alpha_blocked.block_map, bra_alpha_blocked.sort_inds)
    bbc = ci_block_map(bra_beta_blocked.block_map, bra_beta_blocked.sort_inds)
    kac = ci_block_map(ket_alpha_blocked.block_map, ket_alpha_blocked.sort_inds)
    kbc = ci_block_map(ket_beta_blocked.block_map, ket_beta_blocked.sort_inds)

    def reorder_ci(sort_inds, alpha_map, beta_map, ci_coeffs):
        ci_inds = ci_coeffs[sort_inds,:] 
        alpha_sort = alpha_map[sort_inds]
        beta_sort = beta_map[sort_inds]
        return alpha_sort, beta_sort, ci_inds

    # Re-sort bra
    ba_blks, bb_blks, bci = reorder_ci(bra_alpha_blocked.sort_inds, bac, bbc, bra_coeffs)
    # Re-sort ket
    ka_blks, kb_blks, kci = reorder_ci(ket_alpha_blocked.sort_inds, kac, kbc, ket_coeffs)

    mo_ovlps = bra_mos.dot(S_AO).dot(ket_mos.T)
    # Hardcoded:
    #   P = beta
    #   Q = alpha
    # Sort acording to Q (alpha), so P (beta) has to be re-sorted.
    # print("beta ovlps")
    beta_block_ovlps = precompute(bra_beta_blocked, ket_beta_blocked, mo_ovlps)
    beta_block_ovlps[:,-1] = 0.
    # print(beta_block_ovlps)
    # print("alpha ovlps")
    alpha_block_ovlps = precompute(bra_alpha_blocked, ket_alpha_blocked, mo_ovlps)
    alpha_block_ovlps[:,-1] = 0.
    # print(alpha_block_ovlps)

    # Loop over every ket-SD
    wfo = np.zeros((bra_states, ket_states))
    for ket_alpha_block, ket_beta_block, kc in zip(ka_blks, kb_blks, kci):
        # print(ket_alpha_block, ket_beta_block, kc)
        SS = alpha_block_ovlps[ba_blks,ket_alpha_block] * beta_block_ovlps[bb_blks,ket_beta_block]
        dSS = (bci * SS[:,None]).sum(axis=0)
        wfo += np.outer(dSS, kc)

    return wfo


def get_dets(ci_coeffs, ci_thresh=.1):
    occ, virt = ci_coeffs[0].shape
    base_det = list("d" * occ + "e" * virt)
    state, from_, to = np.nonzero(np.abs(ci_coeffs) > ci_thresh)

    spin_adapt = (np.array((1, -1)) * 1/(2**0.5))[:,None]

    # signs = list()
    dets = list()
    indices = list()
    coeffs = list()
    signs = (-1)**(occ - from_ + 1)
    for sign, f, t in zip(signs, from_, to):
        cfs = ci_coeffs[:,f,t] * spin_adapt
        coeffs.extend(sign * cfs)
        t += occ
        # Excitation of alpha; beta left behind
        ab_det = base_det.copy()
        ab_det[f] = "b"
        ab_det[t] = "a"
        dets.append("".join(ab_det))
        indices.append(("a", f, t))
        # Excitation of beta; alpha left behind
        ba_det = base_det.copy()
        ba_det[f] = "a"
        ba_det[t] = "b"
        dets.append("".join(ba_det))
        indices.append(("b", f, t))
    coeffs = np.array(coeffs)
    return dets, indices, coeffs


def get_dets_str(dets, coeffs):
    states = len(coeffs[0])
    det_len = len(dets[0])
    det_num = len(dets)

    header = f"{states} {det_len} {det_num}"
    fmt_ = ["{: >14.10f}", ]
    fmt = " ".join(fmt_ * states)
    lines = [d + fmt.format(*cfs) for d, cfs in zip(dets, coeffs)] 
    det_str = f"{header}\n" + "\n".join(lines)
    return det_str


def expand_det(det):
    """Transform det string into alpha and beta orbital index lists"""

    def trans(ind, char):
        d = {
            "a": (ind, -1),
            "b": (-1, ind),
            "d": (ind, ind),
            "e": (-1, -1),
        }
        return d[char]

    alpha = None
    beta = None

    alpha, beta = zip(*([trans(i, c) for i, c in enumerate(det)])) 
    alpha  = tuple([_ for _ in alpha if _ != -1])
    beta  = tuple([_ for _ in beta if _ != -1])

    return alpha, beta
