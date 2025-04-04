import numpy as np

#1
def gaussian_elimination(AugmentedMatrix):
    n = len(AugmentedMatrix)
    AugmentedMatrix = [row[:] for row in AugmentedMatrix]

    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(AugmentedMatrix[r][i]))
        if abs(AugmentedMatrix[max_row][i]) < 1e-10:
            if any(abs(AugmentedMatrix[max_row][k]) > 1e-10 for k in range(i, n+1)):
                return None
            continue
            
        AugmentedMatrix[i], AugmentedMatrix[max_row] = AugmentedMatrix[max_row], AugmentedMatrix[i]
        pivot = AugmentedMatrix[i][i]
        AugmentedMatrix[i] = [x / pivot for x in AugmentedMatrix[i]]

        for j in range(i+1, n):
            factor = AugmentedMatrix[j][i]
            AugmentedMatrix[j] = [AugmentedMatrix[j][k] - factor * AugmentedMatrix[i][k] for k in range(len(AugmentedMatrix[j]))]

    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = AugmentedMatrix[i][-1] - sum(AugmentedMatrix[i][j] * solution[j] for j in range(i+1, n))

    return solution

#2
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

        L[i, i] = 1

    return L, U

def LU_determinant(A):
    L, U = lu_decomposition(A)
    return np.prod(np.diag(U))

#3
def diagonally_dominant(A):
    n = A.shape[0]
    non_dominant_rows = []

    for i in range(n):
        diagonal = np.abs(A[i, i])
        row_sum = np.sum(np.abs(A[i])) - diagonal
        if diagonal < row_sum:
            non_dominant_rows.append({
                'row': i + 1,
                'diagonal': diagonal,
                'other_sum': row_sum
            })

    if not non_dominant_rows:
        return True, []
    return False, non_dominant_rows

#4
def positive_definite(A):
    if not np.allclose(A, A.T):
        return False, "Matrix is not symmetric"
    
    eigenvals = np.linalg.eigvals(A)
    
    if np.all(eigenvals > 0):
        return True, None
    
    negative_eigenvals = [(i+1, val) for i, val in enumerate(eigenvals) if val <= 0]
    return False, negative_eigenvals