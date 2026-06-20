import numpy as np
from scipy.linalg import expm

def get_CW_matrices(n, Ts):
    """
    Discrete-time Clohessy-Wiltshire-Hill matrices for planar relative
    spacecraft motion, obtained by exact zero-order-hold discretization
    of the continuous-time linearized dynamics at orbital rate n and
    sampling period Ts.
    """
    Ac = np.array([
        [0,      0, 1,    0],
        [0,      0, 0,    1],
        [3*n**2, 0, 0,   2*n],
        [0,      0, -2*n, 0]
    ])
    Bc = np.array([[0,0],[0,0],[1,0],[0,1]])
    M = np.zeros((6, 6))
    M[:4, :4] = Ac
    M[:4, 4:] = Bc
    eM = expm(M * Ts)
    return eM[:4, :4], eM[:4, 4:]
