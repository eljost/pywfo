import numpy as np


SEP = "\n"*3


def mo_ovlp_contraction(shape):
    print("MO Overlaps")
    _ = np.zeros(shape)
    path, string_repr = np.einsum_path("pu,qv,uv->pq",
                                       _, _, _,
                                       optimize="optimal")
    print(path)
    print(string_repr)
    print(SEP)


def coeff_ovlp_contraction(b, k):
    bra_coeffs = np.zeros(b)
    ket_coeffs = np.zeros(k)
    alpha_ovlps = np.zeros((b, k))
    beta_ovlps = np.zeros((b, k))
    path, string_repr = np.einsum_path("b,k,bk,bk",
                                       bra_coeffs, ket_coeffs, alpha_ovlps, beta_ovlps,
                                       optimize="optimal")

    print(path)
    print(string_repr)
    print(SEP)


if __name__ == "__main__":
    dim_ = 4
    mo_ovlp_contraction((dim_, dim_))
    b = 10
    k = 10
    coeff_ovlp_contraction(b, k)
