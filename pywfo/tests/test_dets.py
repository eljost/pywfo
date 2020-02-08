from ..dets import wfoverlap

import h5py
import numpy as np


def test_wfoverlap():
    with h5py.File("ref_cytosin/cytosin_overlap_data.h5") as handle:
        mo_coeffs = handle["mo_coeffs"][:]
        ci_coeffs = handle["ci_coeffs"][:]

    # Compare first and third step 0 and 2
    bra = 0
    ket = 2

    bra_mos = mo_coeffs[bra]
    ket_mos = mo_coeffs[ket]

    ket_inv = np.linalg.inv(ket_mos)
    S_AO = ket_inv.dot(ket_inv.T)

    bra_ci = ci_coeffs[bra]
    ket_ci = ci_coeffs[ket]

    # ci_thresh = 1e-2
    # ci_thresh = 5e-3
    # ci_thresh = 1e-3
    ci_thresh = 7e-2

    wfo = wfoverlap(bra_mos, ket_mos, bra_ci, ket_ci, ci_thresh, S_AO)

    ref = np.array((0.000178, 0.767295, -0.464277, 0.000027)).reshape(-1, 2) 
    np.testing.assert_allclose(wfo, ref, atol=1e-6)

    return wfo
