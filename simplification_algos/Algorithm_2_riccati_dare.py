from algorithm_1_cholesky import cholesky_solve


def zeros(n, m):
    return [[0.0 for _ in range(m)] for _ in range(n)]


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matmul(A, B):
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def mat_add(A, B):
    n, m = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def mat_sub(A, B):
    n, m = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def symmetrize(A):
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def solve_matrix_system(A, B):
    n = len(A)
    m = len(B[0])
    X = zeros(n, m)
    for col in range(m):
        b_col = [B[i][col] for i in range(n)]
        x_col = cholesky_solve(A, b_col)
        for i in range(n):
            X[i][col] = x_col[i]
    return X


def solve_discrete_are(Ad, Bd, Q, R, tol=1e-10, max_iter=500):
    n = len(Ad)
    P = [row[:] for row in Q]

    for it in range(max_iter):
        AdT, BdT = transpose(Ad), transpose(Bd)
        BdT_P = matmul(BdT, P)
        M = symmetrize(mat_add(R, matmul(BdT_P, Bd)))
        BdT_P_Ad = matmul(BdT_P, Ad)
        K_term = solve_matrix_system(M, BdT_P_Ad)

        AdT_P = matmul(AdT, P)
        AdT_P_Ad = matmul(AdT_P, Ad)
        AdT_P_Bd = matmul(AdT_P, Bd)
        correction = matmul(AdT_P_Bd, K_term)

        P_new = symmetrize(mat_sub(mat_add(Q, AdT_P_Ad), correction))

        diff = max(abs(P_new[i][j] - P[i][j]) for i in range(n) for j in range(n))
        P = P_new
        if diff < tol:
            return P, it + 1

    return P, max_iter


def compute_lqr_gain(Ad, Bd, R, P):
    BdT = transpose(Bd)
    BdT_P = matmul(BdT, P)
    M = symmetrize(mat_add(R, matmul(BdT_P, Bd)))
    BdT_P_Ad = matmul(BdT_P, Ad)
    return solve_matrix_system(M, BdT_P_Ad)
