# ___MATRIX MANIPULATIONS____

import numpy as np

#READING IN A MATRIX
def ReadMatrixA (FileName, NumRows, NumColumns):
    A_0 = []
    A_1 = []
    A_3 = []
    A = []
    f = open(FileName, 'r')
    #Reading in all into one array
    for line in f:
        A_0.append(line.rstrip())
    f.close()
    #Making this into a matrix
    for i in range(0,NumRows):
        A_2 = []
        for j in range(0, NumColumns):
            A_2.append(A_0[i*NumColumns+j])
        A_1.append(A_2)
    #Making the elements int instead of strings
    for row in A_1:
        for element in row:
            A_3.append(int(element))
    for i in range(0,NumRows):
        A_4 = []
        for j in range(0, NumColumns):
            A_4.append(A_3[i*NumColumns+j])
        A.append(A_4)
    return A


#TRANSPOSE MATRIX
def Transpose (A):
    #Checking if it is a square matrix
    if len(A) != len(A[0]):
        print(f"{A} is not a square matrix")
        return 0
    #Creating the right sized matrix
    A_T = []
    for i in range(len(A)):
        A_T_ = []
        for j in range(len(A[0])):
            A_T_.append(-1)
        A_T.append(A_T_)

    #Adding the correct values

    for i in range(len(A)):
        for j in range(len(A[0])):
            A_T [i][j] = A[j][i]

    return A_T

#TRACE OF A MATRIX

def Trace(A):
    if len(A) != len(A[0]):
        print(f"{A} is not a square matrix")
        return 0
    T = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if j == i:
                T += A[i][j]
    return T

#MATRIX ADDITION

def MatrixAddition(A, B):

    #Checking if they are the same size
    if not (len(A) == len(B) and len(A[0]) == len(B[0])):
        print("They are different sizes can't add them")

    #Making a right sized result matrix
    C = []
    for i in range(len(A)):
        C_ = []
        for j in range(len(A[0])):
            C_.append(-1)
        C.append(C_)

    #Adding the appropriate values:
    for i in range(len(A)):
        for j in range(len(A[0])):
            C [i][j] = A[i][j] + B[i][j]

    return C


#MATRIX SUBSTRACTION

def MatrixSubstraction(A, B):

    #Checking if they are the same size
    if not (len(A) == len(B) and len(A[0]) == len(B[0])):
        print("They are different sizes can't add them")

    #Making a right sized result matrix
    C = []
    for i in range(len(A)):
        C_ = []
        for j in range(len(A[0])):
            C_.append(-1)
        C.append(C_)

    #Adding the appropriate values:
    for i in range(len(A)):
        for j in range(len(A[0])):
            C [i][j] = A[i][j] - B[i][j]

    return C

#MATRIX MULTIPLICATION

def MatrixMultiplication(A, B):

    #Checking if they are the right size
    if not (len(A[0]) == len(B)):
        print("They are different sizes can't multiply them")

    #Making a right sized result matrix
    C = []
    for i in range(len(A)):
        C_ = []
        for j in range(len(B[0])):
            C_.append(0)
        C.append(C_)


    #Adding the appropriate values:
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C [i][j] += A[i][k]*B[k][j]

    return C

#PRINTING A MATRIX

def PrintMatrix (M):
    for row in M:
            print(row)
    print('_'*40)

 #GAUSSIAN ELIMINATION
    
def GaussianElimination(A,b):
    
    #The number of equations in the system is
    n = len(A)

    #Eliminating the matrix so that it becomes upper triangular form
    for j in range(0,n):
        for i in range(j+1, n):
            #Save p so you can use it even after you changed A (p is the pivot)
            p = (A[i,j] / A[j,j])
            A[i] = A[i] - p * A[j]
            #Adjusting vector b as well
            b[i] = b[i] - p * b[j]
    
    #Finding the values of x_i for each case 
    
    x = np.zeros(n)
    for i in range(n-1,-1, -1):
        x[i] = b[i] / A[i][i]
        for j in range(i+1,n):
                x[i] = x[i] - x[j] * A[i][j] / A[i][i]
    return x


#IMPORTANT: When defining the arrays to use in this function use np.array and set the type to float !!! Lke this
    
    #A = np.array([[8, -2, 1, 3], [1, -5, 2, 1],[-1, 2, 7, 2],[2 ,-1, 3, 8]],dtype=float)
    #b = np.array([9, -7, -1, 5],dtype=float)
