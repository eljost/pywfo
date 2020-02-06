from collections import namedtuple

import h5py
import numpy as np


np.set_printoptions(suppress=True, precision=6, linewidth=120)


def pri(_):
    for __ in _: print(__)


BlockResult = namedtuple(
    "BlockResult", "block_num super_block_num blocks block_map super_map " \
                   "sort_inds ci_coeffs"
)

def block_dets(dets, ci_coeffs):
    # Prefix with indices so we get the actual sorting
    ind_dets = [(i, d) for i, d in enumerate(dets)]
    sort_inds, dets = zip(*sorted(ind_dets, key=lambda x: x[1]))

    # # Blocks
    # blocks = dict.fromkeys(dets)
    # # Superblocks
    # super_blocks = dict.fromkeys([_[:-1] for _ in dets])

    block_num = 1
    super_block_num = 1
    blocks = [list(dets[0]), ]
    block_map = [0, ]
    super_map = [0, ]
    for i, sd in enumerate(dets[1:]):
        # if not (sd == dets[i]).all():
        if not sd == dets[i]:
            block_num += 1
            blocks.append(list(sd))
            # if not (sd[:-1] == dets[i][:-1]).all():
            if not sd[:-1] == dets[i][:-1]:
                super_block_num += 1
                super_map.append(block_num)
        block_map.append(block_num)

    # Sort CI coefficients
    ci_sorted = ci_coeffs[sort_inds,:]

    res = BlockResult(
            block_num,
            super_block_num,
            blocks,
            block_map,
            super_map,
            sort_inds,
            ci_sorted
    )
    return res


def print_dets():
    with h5py.File("tests/ref_cytosin/cytosin_overlap_data.h5") as handle:
        mo_coeffs = handle["mo_coeffs"][:]
        ci_coeffs = handle["ci_coeffs"][:]
    # Compare first and third step 0 and 2
    bra = 0
    ket = 2

    bra_mos = mo_coeffs[bra]
    ket_mos = mo_coeffs[ket]

    ket_inv = np.linalg.inv(ket_mos)
    S_AO = ket_inv.dot(ket_inv.T)
    mo_ovlps = bra_mos.dot(S_AO).dot(ket_mos.T)

    bra_ci = ci_coeffs[bra]
    ket_ci = ci_coeffs[ket]
    
    ci_thresh = 7e-2

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

    print(f"<bra| alpha super-blocks: {bra_alpha_blocked.super_block_num: >16d}")
    print(f"|ket> alpha super-blocks: {ket_alpha_blocked.super_block_num: >16d}")
    print(f"<bra|  beta super-blocks: {bra_beta_blocked.super_block_num: >16d}")
    print(f"|ket>  beta super-blocks: {ket_beta_blocked.super_block_num: >16d}")

    # TODO: signs still missing
    def precompute(bra_blocked, ket_blocked, mo_ovlps):
        block_ovlps = np.zeros((bra_blocked.block_num, ket_blocked.block_num))

        super_map = ket_blocked.super_map
        for i in range(ket_blocked.super_block_num)[:-1]:
            # Compute for all blocks that belong to the given superblock
            cur_start = super_map[i]
            block_slice = slice(cur_start,super_map[i+1])
            for j, ket_block in enumerate(ket_blocked.blocks[block_slice], cur_start):
                for k, bra_block in enumerate(bra_blocked.blocks):
                    ovlp_mat = mo_ovlps[bra_block][:,ket_block]
                    block_ovlps[k,j] = np.linalg.det(ovlp_mat)
        return block_ovlps

    
    beta_block_ovlps = precompute(bra_beta_blocked, ket_beta_blocked, mo_ovlps)
    alpha_block_ovlps = precompute(bra_alpha_blocked, ket_alpha_blocked, mo_ovlps)
    import pdb; pdb.set_trace()
    # return bra_alpha_blocked, bra_beta_blocked, ket_alpha_blocked, ket_beta_blocked


def get_dets(ci_coeffs, ci_thresh=.1):
    occ, virt = ci_coeffs[0].shape
    base_det = list("d" * occ + "e" * virt)
    state, from_, to = np.nonzero(np.abs(ci_coeffs) > ci_thresh)

    spin_adapt = (np.array((1, -1)) * 1/(2**0.5))[:,None]

    dets = list()
    indices = list()
    coeffs = list()
    for f, t in zip(from_, to):
        cfs = ci_coeffs[:,f,t] * spin_adapt
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


def trans(ind, char):
    d = {
        "a": (ind, -1),
        "b": (-1, ind),
        "d": (ind, ind),
        "e": (-1, -1),
    }
    return d[char]

def expand_det(det):
    """Transform det string into alpha and beta orbital index lists"""

    alpha = None
    beta = None

    alpha, beta = zip(*([trans(i, c) for i, c in enumerate(det)])) 
    alpha  = tuple([_ for _ in alpha if _ != -1])
    beta  = tuple([_ for _ in beta if _ != -1])
    # import pdb; pdb.set_trace()

    return alpha, beta


print_dets()
# d = "dddddabeeab"
# ai, bi = expand_det(d)
# import pdb; pdb.set_trace()
# np.testing.assert_allclose(ai, [0,1,2,3,4,5,9])
# np.testing.assert_allclose(bi, [0,1,2,3,4,6,10])
