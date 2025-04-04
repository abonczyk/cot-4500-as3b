import sys
import os
import numpy as np

sys.path.append(os.path.abspath('C:/Users/andre/Desktop/repo_4/src/main'))

from assignment_3 import gaussian_elimination, lu_decomposition, LU_determinant, diagonally_dominant, positive_definite

#1
augmented_matrix = [
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
]
solution = gaussian_elimination(augmented_matrix)
if solution is None:
    print("The system has no solution")
else:
    print("Solution: ", solution)

#2
A = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)

L, U = lu_decomposition(A)
det_A = LU_determinant(A)
print(f"\nDeterminant: {det_A}")
print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)


#3
A_diag = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
], dtype=float)

result, details = diagonally_dominant(A_diag)
if result:
    print("The matrix is diagonally dominant.")
else:
    print("The matrix is not diagonally dominant.")
    print("\nSpecific issues:")
    for row in details:
        print(f"Row {row['row']}: |diagonal| ({row['diagonal']:.2f}) < sum of other elements ({row['other_sum']:.2f})")

#4
A_pd = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)

result, details = positive_definite(A_pd)
if result:
    print("\nThe matrix is positive definite.")
else:
    print("\nThe matrix is NOT positive definite.")
    if isinstance(details, str):
        print(f"Reason: {details}")
    else:
        print("\nSpecific issues:")
        for idx, val in details:
            print(f"Eigenvalue {idx} is not positive: {val:.4f}")