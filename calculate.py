import numpy as np 

def mat_multiple(mat1, mat2):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return np.dot(mat1, mat2)
    
mat1 = [[-np.sqrt(3)/2, 1/2], [-1/2, -np.sqrt(3)/2]]
mat2 = [[3, 0], [0, 5]]
mat3 = [[1 / np.sqrt(2), 1 / np.sqrt(2)], [- 1 / np.sqrt(2), 1 / np.sqrt(2)]]

mat = mat_multiple(mat1, mat_multiple(mat2, mat3))

mat4 = [[-1 / np.sqrt(5), 2 / np.sqrt(5)], [-2 / np.sqrt(5), -1 / np.sqrt(5)]]
mat5 = [[2, 0], [0, 0]]
x = [-1 / np.sqrt(2), 1 / np.sqrt(2)]
mat = mat_multiple(mat_multiple(mat4, mat5), mat_multiple(mat3, x))
print(mat)
print((-3 * np.sqrt(3) + 5) / (np.sqrt(2) * 2))