class VectorMatrixOperations:
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)
    def __repr__(self):
        return repr(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        return self.data[key]
    def __add__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations(list(map(sum, zip(self.data, other.data))))
        else:
            return VectorMatrixOperations([x + other for x in self.data])

    def __sub__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations(list(map(lambda lists: lists[0] - lists[1], zip(self.data, other.data))))
        else:
            return VectorMatrixOperations([x - other for x in self.data])

    def __mul__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations([a*b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([x * other for x in self.data])
    def __rmul__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations([a*b for a,b in zip(other.data, self.data)])
        else:
            return VectorMatrixOperations([other * x  for x in self.data])
    def __truediv__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations([a/b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([x / other for x in self.data])
    def __rtruediv__(self, other):
        if isinstance(other, VectorMatrixOperations):
            return VectorMatrixOperations([a/b for a,b in zip(other.data, self.data)])
        else:
            return VectorMatrixOperations([other / x  for x in self.data])

    def __pow__(self, power):
        return VectorMatrixOperations([num ** power for num in self.data])

    def Transpose(self):
        return VectorMatrixOperations([list(x) for x in zip(*self.data)])
 
def dotProduct(v1, v2):
    return sum(map(lambda lists: lists[0] * lists[1], zip(v1, v2)))
def dotProductMatrix(matrix2d, matrix1d):
    return VectorMatrixOperations([dotProduct(row, matrix1d) for row in matrix2d])           
def mutiply_element_wise(matrix1, matrix2):
    return VectorMatrixOperations([a*b for a,b in zip(matrix1, matrix2)])
def matrix_2d_1d_multiply(matrix_2_d, matrix_1_d):
    return [mutiply_element_wise(row, matrix_1_d) for row in matrix_2_d]    
def transpose_2dmatrix(matrix):
    return [list(i) for i in zip(*matrix)]
def mean(matrix, axis=None):
    if axis == None:
        return sum(matrix) / len(matrix)
    elif axis == 0:
        return [sum(col) / len(col) for col in transpose_2dmatrix(matrix)]
    elif axis == 1:
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Invalid axis")