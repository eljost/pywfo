import os
from pathlib import Path

import h5py
import numpy as np


from .test_ovlps import perturb_mat
from ..main import moovlp, moovlp_expl, moovlp_dots

def moovlp(mos1, mos2, S_AO):
    """Overlap between two sets of MOs.

    <phi_p|phi_q> = sum_{u,v} C_{pu} C'_{qv} <X_u|X'_v>
    """

    # S_AO == None will only work if mos1 are comprise all MOs (occ. + virt.).
    # If we only supply a subset of MOs like in an excited state SD then this
    # will not work. So I deactivate it for now.
    # if S_AO is None:
        # mos1_inv = np.linalg.inv(mos1)
        # S_AO = mos1_inv.dot(mos1_inv.T)

    ovlp = np.einsum("pu,qv,uv->pq", mos1, mos2, S_AO,
                     optimize=['einsum_path', (0, 2), (0, 1)])
    return ovlp

np.set_printoptions(suppress=True, precision=6)

THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

def test_moovlps():
    # with h5py.File(THIS_DIR / "ref_cytosin/cytosin_overlap_data.h5") as handle:
        # mo_coeffs = handle["mo_coeffs"][:]
    # mi = np.linalg.inv(mo_coeffs[0])
    # SAO = mi.dot(mi.T)
    # mo0 = mo_coeffſ[0]
    # mo1 = mo_coeffſ[1]

    # [moovlp(mo0, mo1, SAO) for i in range(25000)]
    np.random.seed(20180325)
    d = 10
    dd = (d, d)
    mo0 = np.random.rand(*dd)
    mo0, _ = np.linalg.qr(mo0, mode="complete")
    mo0 = mo0.T

    mo1, _ = np.linalg.qr(perturb_mat(mo0.T), mode="complete")
    mo1 = mo1.T
    mo0i = np.linalg.inv(mo0)
    S_AO = mo0i.dot(mo0i.T)
    # import pdb; pdb.set_trace()
    ovlp = moovlp(mo0, mo1, S_AO)
    ovlp_expl = moovlp_expl(mo0, mo1, S_AO)
    ovlp_dots = moovlp_dots(mo0, mo1, S_AO)
    np.testing.assert_allclose(ovlp_expl, ovlp)
    np.testing.assert_allclose(ovlp_dots, ovlp)
