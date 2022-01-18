#___NUMERICAL INTERPOLATION___

import numpy as np

#NOTE: The cubic splines will require the gaussian elimination fuction from the Matrix manipulations

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



#LAGRANGIAN INTEROPOLATION (USES BOTH FUNCTIONS BELOW)
    
def Lagrangian(j, xp, xn):
    Lj = 1
    for k in range(len(xn)):
        if k != j:
            Lj = Lj * (xp - xn[k]) / (xn[j] - xn[k])
        else:
            pass
    return Lj

def LagrangianInterpolation(xn, yn, x):
    y = []
    for i in range(len(x)):
        a = 0
        xp = x[i]
        for j in range(len(yn)):
            a += yn[j] * Lagrangian(j,xp, xn)
        y.append(a)
    return y

#NEWTON FORWARD DEVIDED DIFFERENCE (USES BOTH FUNCTIONS BELOW)
    
#Defining the function that will come up with a_i from the experession. (This will take it to the base case every time, but by giving it different inputs it we will be able to come to "different levels")

def newtonDevidedDifference(xn, yn):
    if len(xn) == 1:
        return yn[0]
    else:
        return (newtonDevidedDifference(xn[1:],yn[1:]) - newtonDevidedDifference(xn[0:-1],yn[0:-1])) / (xn[-1]-xn[0])
    
#Defining the function that will do the actual interpolation for us NOTE: USES THE GAUSSIAN ELIMINATION FUNCTION:
    
def newtonInterpolation(xn, yn, x):
    
    y = [] #making the target array for the interpolation values
    
    for k in range(len(x)):
        y_ = 0
        for i in range(len(xn)):
            y__ = newtonDevidedDifference(xn[0:i+1],yn[0:i+1])
            for j in range(i):
                y__ = y__ * (x[k] - xn[j])
            y_ += y__
        y.append(y_)
    return y

#Creating cubic splines with the clamped boundary conditions and then interpolating that on a given array of x:
    
def splines(xn,yn, boundaryConditionLower, boundaryConditionUpper, x):
    
    #Create the arrays into which the coefficients will go:
    
    aj = np.ndarray(len(xn)-1)
    bj = np.ndarray(len(xn)-1)
    cj = np.ndarray(len(xn)-1)
    dj = np.ndarray(len(xn)-1)
    
    #Create the matrix A and Collumn vector b that will help find the first derivatives v_j-s

    A = np.zeros((len(xn),len(xn)))
    b = np.zeros(len(xn))
    
    #Inputing the information we know into these matricies:
    b[0] = boundaryConditionLower
    b[-1] = boundaryConditionUpper
    
    A[0,0]   = 1
    A[-1,-1] = 1
    
    #Filling up the rest of the matrix and collumn vector:
    
    for i in range(1,len(xn)-1):
        A[i,i-1] = 1 / (xn[i]-xn[i-1])
        A[i,i] = 2 / (xn[i]-xn[i-1]) + 2 / (xn[i+1]-xn[i])
        A[i,i+1] = 1 / (xn[i+1]-xn[i])

        b[i] = 3 * ((yn[i]-yn[i-1]) / (xn[i]-xn[i-1])**2 + (yn[i+1]-yn[i]) / (xn[i+1]-xn[i])**2 )

    
    #Solving the resulting system of linear equations:
    
    v = GaussianElimination(A,b)
    
    #Determining the coefficients using all this:
    
    for i in range(len(xn)-1):
        aj[i] = yn[i]
        bj[i] =  v[i]
        cj[i] = 3*(yn[i+1]-yn[i])/(xn[i+1]-xn[i])**2 - (v[i+1]+2*v[i])/(xn[i+1]-xn[i])
        dj[i] = -2 * (yn[i+1] - yn[i]) / (xn[i+1] - xn[i]) ** 3 + (v[i+1] + v[i]) / (xn[i+1] - xn[i]) ** 2
        
    
    #Interpolate with these:
    
    y = np.zeros(len(x))
    
    for j in range(len(xn)-1):
        y[(xn[j]<=x) & (x<=xn[j+1])] = aj[j] + bj[j]*(x[(xn[j]<=x) & (x<=xn[j+1])]-xn[j]) +  \
              cj[j]*(x[(xn[j]<=x) & (x<=xn[j+1])]-xn[j])**2 + dj[j]*(x[(xn[j]<=x) & (x<=xn[j+1])]-xn[j])**3
        
    return y




