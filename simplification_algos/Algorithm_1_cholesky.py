def zeros(n, m):
    return [[0.0 for _ in range(m)] for _ in range(n)]


def cholesky(A):
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = (val if val > 1e-15 else 1e-15) ** 0.5
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def cholesky_solve(A, b):
    # Solves A @ x = b for A symmetric positive definite
    L = cholesky(A)
    y = forward_substitution(L, b)
    LT = transpose(L)
    return backward_substitution(LT, y)
