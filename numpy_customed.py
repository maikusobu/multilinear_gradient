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
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):  
                return VectorMatrixOperations([[a + b for a, b in zip(row, other.data)] for row in self ])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):  
               return VectorMatrixOperations([[a + b for a, b in zip(row, self.data)] for row in other ])
            else:
               return VectorMatrixOperations([a + b for a, b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([x + other for x in self])
    def __sub__(self, other):
        if isinstance(other, VectorMatrixOperations):
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):  
                return VectorMatrixOperations([[a - b for a, b in zip(row, other.data)] for row in self])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):
                return VectorMatrixOperations([[a - b for a, b in zip(row, self.data)] for row in other])
            else:  
                return VectorMatrixOperations(list(map(lambda lists: lists[0] - lists[1], zip(self.data, other.data))))
        else:
            return VectorMatrixOperations([x - other for x in self])
    def __mul__(self, other):
        if isinstance(other, VectorMatrixOperations):
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):   
                return VectorMatrixOperations([[a * b for a, b in zip(row, other.data)] for row in self])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):  
                return VectorMatrixOperations([[a * b for a, b in zip(row, self.data)] for row in other])
            else:  
                return VectorMatrixOperations([a*b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([x * other for x in self])

    def __truediv__(self, other):
        if isinstance(other, VectorMatrixOperations):
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):  
                return VectorMatrixOperations([[a / b for a, b in zip(row, other.data)] for row in self])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):  
                return VectorMatrixOperations([[a / b for a, b in zip(row, self.data)] for row in other])
            else:  
                return VectorMatrixOperations([a/b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([x / other for x in self.data])
    def __rmul__(self, other):
        if isinstance(other, VectorMatrixOperations):
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):  
                return VectorMatrixOperations([[a * b for a, b in zip(row, other.data)] for row in self])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):  
                return VectorMatrixOperations([[a * b for a, b in zip(row, self.data)] for row in other])
            else:  
                return VectorMatrixOperations([a*b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([other * x for x in self])
    def __rtruediv__(self, other):
        if isinstance(other, VectorMatrixOperations):
            if isinstance(self.data[0], list) and not isinstance(other.data[0], list):  
                return VectorMatrixOperations([[a / b for a, b in zip(row, other.data)] for row in self])
            elif isinstance(other.data[0], list) and not isinstance(self.data[0], list):  
                return VectorMatrixOperations([[a / b for a, b in zip(row, self.data)] for row in other])
            else:  
                return VectorMatrixOperations([a/b for a,b in zip(self.data, other.data)])
        else:
            return VectorMatrixOperations([other / x for x in self])
    def __pow__(self, power):
        if isinstance(self.data[0], list):  
            return VectorMatrixOperations([[num ** power for num in row] for row in self])
        else:  
            return VectorMatrixOperations([num ** power for num in self])

    def Transpose(self):
        if isinstance(self.data[0], list): 
            return VectorMatrixOperations([list(x) for x in zip(*self.data)])
        else:  
            return VectorMatrixOperations(self.data)
def dotProduct(v1, v2):
    if isinstance(v1, VectorMatrixOperations) and isinstance(v2, VectorMatrixOperations):
        if isinstance(v1.data[0], list) and len(v1.data[0]) == len(v2.data): 
            return VectorMatrixOperations([dotProduct(row, v2.data) for row in v1.data])
        elif isinstance(v2.data[0], list) and len(v2.data[0]) == len(v1.data):  
            return VectorMatrixOperations([dotProduct(v1.data, row) for row in v2.data])
        elif not isinstance(v1.data[0], list) and not isinstance(v2.data[0], list): 
            return sum(map(lambda lists: lists[0] * lists[1], zip(v1.data, v2.data)))
        else:
            raise ValueError("ValueError")
    elif isinstance(v1, list) and isinstance(v2, list):
        return sum(map(lambda lists: lists[0] * lists[1], zip(v1, v2)))
    else:
        raise TypeError("TypeError")
def mean(matrix: VectorMatrixOperations, axis=None):
    if axis == None:
        return sum(matrix) / len(matrix)
    elif axis == 0:
        return VectorMatrixOperations([sum(col) / len(col) for col in matrix.Transpose()])
    elif axis == 1:
        return VectorMatrixOperations([sum(row) / len(row) for row in matrix])
    else:
        raise ValueError("Invalid axis")