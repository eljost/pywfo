from collections import namedtuple

import numpy as np


BlockResult = namedtuple(
    # "BlockResult", "block_num super_block_num blocks block_map super_map " \
                   # "sort_inds ci_coeffs"
    "BlockResult", "block_num super_block_num blocks " \
                   "block_map super_map ci_block_map " \
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

    ci_block_map = np.zeros_like(block_map)
    ci_block_map[sort_inds] = block_map

    res = BlockResult(
            block_ind+1,
            super_block_ind+1,
            blocks,
            block_map,
            super_map,
            ci_block_map,
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


def prepare_data(ci_coeffs, n_alpha, n_beta, ci_thresh, det_fn=None):
    signs, dets, inds, coeffs = get_dets(ci_coeffs, ci_thresh)

    prefacts, alpha_dets, beta_dets = zip(*[expand_det_string(det, n_alpha, n_beta)
                                            for det in dets])
    prefacts = np.array(prefacts)
    coeffs *= prefacts[:,None]# * signs[:,None]
    alpha_blocked = block_dets(alpha_dets, coeffs)
    beta_blocked = block_dets(beta_dets, coeffs)

    if det_fn:
        det_str = get_dets_str(dets, coeffs)
        with open(det_fn, "w") as handle:
            handle.write(det_str)

    return coeffs, alpha_blocked, beta_blocked


def precompute(bra_blocked, ket_blocked, mo_ovlps):
    block_ovlps = np.zeros((bra_blocked.block_num, ket_blocked.block_num))

    for i, bra_block in enumerate(bra_blocked.blocks):
        for j, ket_block in enumerate(ket_blocked.blocks):
            ovlp_mat = mo_ovlps[bra_block][:,ket_block]
            block_ovlps[i,j] = np.linalg.det(ovlp_mat)
    return block_ovlps


def wfoverlap(bra_mos, ket_mos, bra_ci, ket_ci, ci_thresh, S_AO, closed_shell=True):
    # Right now only closed shell is supported
    assert closed_shell
    # Assert same number of MOs for bra and ket
    np.testing.assert_allclose(bra_ci.shape[1:], ket_ci.shape[1:])

    n_alpha = bra_ci.shape[1]
    n_beta = n_alpha

    bra_states = bra_ci.shape[0]
    ket_states = ket_ci.shape[0]

    def prep(ci_coeffs):
        return prepare_data(ci_coeffs, n_alpha, n_beta, ci_thresh)

    bra_coeffs, bra_alpha_blocked, bra_beta_blocked = prep(bra_ci)
    ket_coeffs, ket_alpha_blocked, ket_beta_blocked = prep(ket_ci)

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

    print(f"<bra| alpha super-blocks: {bra_alpha_blocked.super_block_num: >16d}")
    print(f"|ket> alpha super-blocks: {ket_alpha_blocked.super_block_num: >16d}")
    print(f"<bra|  beta super-blocks: {bra_beta_blocked.super_block_num: >16d}")
    print(f"|ket>  beta super-blocks: {ket_beta_blocked.super_block_num: >16d}")

    # print("bra alpha")
    # dbg_print(bra_alpha_blocked, bra_coeffs)
    # print("bra beta")
    # dbg_print(bra_beta_blocked, bra_coeffs)
    # print("ket alpha")
    # dbg_print(ket_alpha_blocked, ket_coeffs)
    # print("ket beta")
    # dbg_print(ket_beta_blocked, ket_coeffs)

    bac = bra_alpha_blocked.ci_block_map
    bbc = bra_beta_blocked.ci_block_map
    kac = ket_alpha_blocked.ci_block_map
    kbc = ket_beta_blocked.ci_block_map

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
    # print(beta_block_ovlps)
    # print("alpha ovlps")
    alpha_block_ovlps = precompute(bra_alpha_blocked, ket_alpha_blocked, mo_ovlps)
    # print(alpha_block_ovlps)

    # Loop over every ket-SD
    wfo = np.zeros((bra_states, ket_states))
    for ket_alpha_block, ket_beta_block, kc in zip(ka_blks, kb_blks, kci):
        SS = alpha_block_ovlps[ba_blks,ket_alpha_block] * beta_block_ovlps[bb_blks,ket_beta_block]
        dSS = (bci * SS[:,None]).sum(axis=0)
        wfo += np.outer(dSS, kc)

    return wfo


def get_dets(ci_coeffs, ci_thresh=.1):
    occ, virt = ci_coeffs[0].shape
    base_det = list("d" * occ + "e" * virt)
    state, from_, to = np.nonzero(np.abs(ci_coeffs) > ci_thresh)

    sa_factor = (np.array((1, -1))/(2**0.5))[:,None]

    dets = list()
    indices = list()
    coeffs = list()

    signs = np.repeat((-1)**(occ - from_ + 1), 2)

    for sign, f, t in zip(signs, from_, to):
        cfs = ci_coeffs[:,f,t] * sa_factor
        # coeffs.extend(sign * cfs)
        coeffs.extend(cfs)
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
    return signs, dets, indices, coeffs


def expand_det_string(det_str, nalpha, nbeta):
    # Number of alpha (beta) electrons already found
    na = 0
    nb = 0
    # Number of alpha electrons reamining to be found
    permutations = 0
    # Alpha/beta SDs holding the corresponding orbital indices
    alpha_inds = list()
    beta_inds = list()
    # alpha_det = [0 for np.zeros(nalpha, dtype=int)
    # beta_det = np.zeros(nbeta, dtype=int)

    # The overlap of two Slater-determinants is given by the determinant
    # of the overlap matrix of the spin orbitals that make up the two SDs.
    # When the SDs are sorted in a way that alpha spin orbitals appear before
    # beta spin orbitals the determinant of the overlap matrix becomes block
    # diagonal, so we may have to resort the determinant/permute the orbitals.
    #
    # Whenever we find a beta-electron but there are still alpha-electrons
    # to be found we have to do permuations. E.g. if we find a beta electron
    # and there are still 4 alpha electrons left we have to do four permuations
    # to bring the beta electron to the end of the determinant.
    #   0: baaaa
    #   --start--
    #   1: abaaa
    #   2: aabaa
    #   3: aaaba
    #   4: aaaab
    #   --sorted after 4 permutations--

    for i, char in enumerate(det_str):
        # Alpha spin orbital
        if char == "a":
            # alpha_det[na] = i
            alpha_inds.append(i)
            na += 1
        # Beta spin orbital
        elif char == "b":
            # beta_det[nb] = i
            beta_inds.append(i)
            permutations += nalpha - na
        # Doubly occupied, alpha and beta spin orbitals
        elif char == "d":
            # alpha_det[na] = i
            # beta_det[nb] = i
            alpha_inds.append(i)
            beta_inds.append(i)
            na += 1
            nb += 1
            permutations += nalpha - na
        # Empty
        elif char == "e":
            continue
        else:
            raise Exception( "Invalid det string. Expected one of 'deab' "
                            f"but got {char} at index {i}!")
    # Permuting two rows or two columns of a determinant changes its sign.
    prefactor = (-1)**permutations

    # return prefactor, alpha_det, beta_det
    return prefactor, alpha_inds, beta_inds


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
