import numpy as np


SEP = "\n"*3


def mo_ovlp_contraction(shape):
    print("MO Overlaps")
    _ = np.zeros(shape)
    path, string_repr = np.einsum_path("pu,qv,uv->pq", _, _, _,
                                       optimize="optimal")
    print(path)
    print(string_repr)
    print(SEP)


if __name__ == "__main__":
    dim_ = 4
    mo_ovlp_contraction((dim_, dim_))
