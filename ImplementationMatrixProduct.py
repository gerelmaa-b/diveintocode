import numpy as np 

a_ndarray = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
b_ndarray = np.array([[0, 2, 1], [0, 2, -8], [2, 9, -1]])

d_ndarray = np.array([[-1, 2, 3], [4, -5, 6]])
e_ndarray = np.array([[-9, 8, 7], [6, -5, 4]])

################ Problem 2 ##################
c_ndarray = np.matmul(a_ndarray, b_ndarray)
c_ndarray_dot = np.dot(a_ndarray, b_ndarray)
c_ndarray_at = a_ndarray @ b_ndarray

print("matrix multiplication by np.matmul: {}".format(c_ndarray))
print("matrix multiplication by np.dot: {}".format(c_ndarray_dot))
print("matrix multiplication by np.dot: {}".format(c_ndarray_at))

################ Problem 3, 4, 5 ###################
# Creation of a function that performs matrix multiplication

def matrix_multiplication(a_ndarray, b_ndarray):
    '''
    This function is calculate product of two matrices
    Parameters 
    ------------------
    a_ndarray : ndarray
        a matrix to multiply
    b_ndarray : ndarray
        b matrix to multiply
    
    Returns
    -------------------
    c_ndarray_man : ndarray
        c = a * b
    '''
    c_ndarray_man=np.empty((a_ndarray.shape[0], b_ndarray.shape[1]))
    print("a:", a_ndarray.shape[1])
    print("b:", b_ndarray.shape[0])
    for i in range(a_ndarray.shape[0]):
        for j in range(b_ndarray.shape[1]):
            for k in range(b_ndarray.shape[1]):
                c_ndarray_man[i,j] = c_ndarray_man[i,j] + a_ndarray[i,k] * b_ndarray[k,j]
    
    return c_ndarray_man

if d_ndarray.shape[0] == e_ndarray.shape[1]:
    c_ndarray_man = matrix_multiplication(d_ndarray, e_ndarray)
else:
    print((np.transpose(e_ndarray)).shape)
    print(d_ndarray.shape)
    c_ndarray_man = matrix_multiplication(d_ndarray, np.transpose(e_ndarray))
    print("c_ndarray_man:", c_ndarray_man)
    #raise Exception('e and c matrix must be same column and row. d matrix column was: {}, e matrix row was: {}'.format(d_ndarray.shape[1], e_ndarray.shape[0]))

print("matrix multiplication by manual: {}".format(c_ndarray_man))





