import numpy as np

def ci_coeffs_above_thresh(ci_coeffs, thresh=1e-5):
    mo_inds = np.where(np.abs(ci_coeffs) > thresh)
    return mo_inds

def make_det_string(inds):
    """Return spin adapted strings."""
    from_mo, to_mo = inds
    # Until now the first virtual MO (to_mo) has index 0. To subsitute
    # the base_str at the correct index we have to increase all to_mo
    # indices by the number off occupied MO.
    to_mo += occ_mo_num
    # Make string for excitation of an alpha electron
    ab = list(base_det_str)
    ab[from_mo] = "b"
    ab[to_mo] = "a"
    ab_str = "".join(ab)
    # Make string for excitation of an beta electron
    ba = list(base_det_str)
    ba[from_mo] = "a"
    ba[to_mo] = "b"
    ba_str = "".join(ba)
    return ab_str, ba_str

def generate_all_dets(occ_set1, virt_set1, occ_set2, virt_set2):
    """Generate all possible single excitation determinant strings
    from union(occ_mos) to union(virt_mos)."""
    # Unite the respective sets of both calculations
    occ_set = occ_set1 | occ_set2
    virt_set = virt_set1 | virt_set2
    # Genrate all possible excitations (combinations) from the occupied
    # MO set to (and) the virtual MO set.
    all_inds = [(om, vm) for om, vm
                in itertools.product(occ_set, virt_set)]
    det_strings = [make_det_string(inds) for inds in all_inds]
    return all_inds, det_strings

def make_full_dets_list(all_inds, det_strings, ci_coeffs):
    dets_list = list()
    for inds, det_string in zip(all_inds, det_strings):
        ab, ba = det_string
        from_mo, to_mo = inds
        per_state =  ci_coeffs[:,from_mo,to_mo]
        # Drop unimportant configurations, that are configurations
        # having low weights in all states under consideration.
        if np.sum(per_state**2) < conf_thresh:
            continue
        # A singlet determinant can be formed in two ways:
        # (up down) (up down) (up down) ...
        # or
        # (down up) (down up) (down up) ...
        # We take this into account by expanding the singlet determinants
        # and using a proper normalization constant.
        # See 10.1063/1.3000012 Eq. (5) and 10.1021/acs.jpclett.7b01479 SI
        per_state *= 1/2**0.5
        as_str = lambda arr: " ".join([fmt.format(cic)
                                       for cic in arr])
        ps_str = as_str(per_state)
        mps_str = as_str(-per_state)
        dets_list.append(f"{ab}\t{ps_str}")
        dets_list.append(f"{ba}\t{mps_str}")
    return dets_list


def get_from_to_sets(ci_coeffs):
    all_mo_inds = [ci_coeffs_above_thresh(per_state)
                   for per_state in ci_coeffs]

    from_mos, to_mos = zip(*all_mo_inds)
    from_set = set_from_nested_list(from_mos)
    to_set = set_from_nested_list(to_mos)

    return from_set, to_set


def wf_overlap(cycle1, cycle2, ao_ovlp=None):
    mos1, cic1 = cycle1
    mos2, cic2 = cycle2

    fs1, ts1 = get_from_to_sets(cic1)
    fs2, ts2 = get_from_to_sets(cic2)

    # Create a fake array for the ground state where all CI coefficients
    # are zero and add it.
    all_inds, det_strings = generate_all_dets(fs1, ts1, fs2, ts2)
