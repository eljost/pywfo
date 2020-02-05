import numpy as np

def laplace(mat, expcol=-1):
    if mat.shape == (1, 1):
        return mat[0,0]
    elif mat.shape == (2, 2):
        a, b, c, d = mat.flatten()
        return a*d - b*c

    rows, cols = mat.shape
    det = 0.
    for i in range(rows):
        #                           rows above i,     rows below i
        minor = np.concatenate((mat[:i, :expcol], mat[i+1:, :expcol]))
        det += (-1)**(i+1+cols) * mat[i,-1] * laplace(minor)
    return det


mat = np.arange(9).reshape(-1 ,3) + 1
import pdb; pdb.set_trace()
det = laplace(mat, 2)
print(det)
assert  det == 0
