import numpy as np

from pywfo.main import moovlp, moovlp_expl, moovlp_dots
from pywfo.helpers import perturb_mat


def test_moovlps():

    # Set up fake MOs
    np.random.seed(20180325)
    mo0 = np.random.rand(*(10, 10))
    mo0, _ = np.linalg.qr(mo0, mode="complete")
    mo0 = mo0.T

    # Slightly perturb original matrix
    mo1, _ = np.linalg.qr(perturb_mat(mo0.T), mode="complete")
    mo1 = mo1.T

    # Determine AO overlaps
    mo0i = np.linalg.inv(mo0)
    S_AO = mo0i.dot(mo0i.T)

    ovlp = moovlp(mo0, mo1, S_AO)
    ovlp_expl = moovlp_expl(mo0, mo1, S_AO)
    ovlp_dots = moovlp_dots(mo0, mo1, S_AO)
    np.testing.assert_allclose(ovlp_expl, ovlp)
    np.testing.assert_allclose(ovlp_dots, ovlp)
