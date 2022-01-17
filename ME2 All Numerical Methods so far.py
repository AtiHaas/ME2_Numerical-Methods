#CID: 1857314


#____LIBRARIES YOU MIGHT NEED____

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits import mplot3d

#____NUMERICALLY REPRESENTING FUCTIONS____

#Discretiseing axies:

def discretiseAxis1(x_0, x_n, numNodes): #Discretises the axis in the range defined by x_0 and x_n to numNodes number of nodes. (No need to calculate the step size)
    x = np.linspace(x_0,x_n, numNodes)
    return x


def discretiseAxis2(x_0, x_n, stepSize): #Discretises the axis in the range defined by x_0 and x_n with a step size of stepSize. (No need to calcutae the number of nodes)
    x = np.arange(x_0, x_n, stepSize)
    return x

#GENERATING MESHES:
    
#For f(x,y):
    
def makeMesh(x_0, x_n, y_0, y_n, stepSize):
    #Defining the  spacial domains of both axies and discretiseing them:
    x = np.arange(x_0, x_n, stepSize)
    y = np.arange(y_0, y_n, stepSize) 

    #Creating the grids:
    (Xg, Yg) = np.meshgrid(x,y) #This function creates the 2d grids as needed
    return Xg, Yg #IMPORTANT: Use numpy library for the math functions when working with these grids (Only this will do it mesh wise)

#For f(x,y,t)

def makeMeshWithTime(x_0, x_n, y_0, y_n, t_0, t_n, stepSizeXY, stepSizeT):
    #Defining the spacial domains of both axies and discretiseing them:
    x = np.arange(x_0, x_n, stepSizeXY)
    y = np.arange(y_0, y_n, stepSizeXY) 
     #Defining the time domain and discretiseing it:
    t = np.arange(t_0, t_n, stepSizeT)
    
    #Creating the grids:
    (Xg, Yg, Tg) = np.meshgrid(x, y, t)
    
    return Xg, Yg, Tg
    
    


#FINDING THE "LENGTHS IN ALL DIRECTIONS" OF MESHES:

def lenGrid(Xg): 
    print(np.shape(Xg))
    
    
#PLOTTING RESULTS IN 3D:
    
def threeDPlot(Xg, Yg, Surface): #This funciton is for a grid that describes a f(x,y)
    
    #Plotting S
    print('S(x,y) is the following in 3D:')
    ax = pl.axes(projection='3d')
    ax.plot_surface(Xg,Yg,Surface)
    
    #Plotting the contour of S
    pl.show()           # make a new window plot
    print('S(x,y) has the following contour plot:')
    pl.contour(Xg,Yg,Surface) # plot contours
    
def threeDPlotAtTimeT(Xg, Yg, Tg, Surface, atT): #This funciton is for a grid that describes a f(x,y,t)
    print(f'The 3D plot of R at t={atT}s will be:')
    ax = pl.axes(projection='3d')
    ax.plot_surface(Xg[:,:,atT], Yg[:,:,atT], Surface[:,:,atT])
    
    
    #MAKING A 2D VECTOR SPACE

def twoDVectorInAPlane(x_0, x_n, y_0, y_n, stepSize):
    #Defining the  spacial domains of both axies and discretiseing them:
    x = np.arange(x_0, x_n, stepSize)
    y = np.arange(y_0, y_n, stepSize) 
    #String their lengths:
    LenX = len(x)
    LenY = len(y)

    #Creating the mesh for the plane
    Xg, Yg = np.meshgrid(x,y)
    
    # allocate an array for the vector field F: size is Nx by Ny by 2 (2 is because the vector has two components, i and j)
    F = np.ndarray((LenX,LenY, 2))
    
    #Calculating the values of the vector field (!Change the expressions!):
    F[:,:,0] = Xg #i-component, WATCH OUT THE INDEXING IS FROM 0
    F[:,:,1] = Yg #j-component
    
#PLOTTING A 2D VECTOR SPACE

def plot2DVectorSpace(Xg, Yg, F):
    #Plotting the values:

    print('The vector field in Problem a), plotted, looks like the following:')
    pl.quiver(Xg, Yg, F[:,:,0], F[:,:,1])

    pl.show()
    print('The vector field in Problem a), plotted as streamlines, looks like the following:')
    pl.streamplot(Xg, Yg, F[:,:,0], F[:,:,1])
   
    
    
 #____GAUSSIAN ELIMINATION____
    
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

 #____READING DATA FROM FILES____

#READING IN EACH LINE INTO AN ARRAY
def ReadFileToArray (FileName):
    f = open(FileName, "r")
    FileArr = []
    for line in f:
        FileArr.append(float(line.rstrip())) #In case you need int modigfy this line, float is the most general so I went with that one here :D
    f.close()
    return FileArr

 #____NUMERICAL METHODS____

#FORWARDS NUMERICAL DIFFERENTATION

def forwardsDifferentiation(xn,yn):
    
    dydx = []
    
    for i in range(len(xn)-1):
        dydx.append((yn[i+1] - yn[i]) / (xn[i+1] - xn[i]))
        
    return dydx

#FORWARDS NUMERICAL DIFFERENTATION K TIMES (THIS USES THE FUNCTION ABOVE) NOTE: THIS IS VERY EXPENSIVE COMPUTATIONALLY BUT WILL WORK, THE BETTER VERSION BELOW WILL USE THE BINOMIAL SERIES TO QUICKEN THINGS IF THE STEP SIZE IS EQUAL EVERYWHERE

def forwardsDifferentiationKTimes(xn,yn, k):
    dydx = forwardsDifferentiation(xn,yn)
    for i in range(1, k):
        dydx = forwardsDifferentiation(xn[:-i], dydx) #The x-s have to be split since we lose a node with each order of differentiation
    return dydx

#FORWARDS NUMERICAL DIFFERENTATION K TIMES EFFICIENT WAY WITH EQUIDISTANT NODES
#We will be using binomial coefficients to speed up the proscess and this way we will be able to jump to the right dervivative at first:
#For this we will need a few functions that help to find the coefficients

def factorial(n):
    if n==0:
        return 1
    else:
        return n*factorial(n-1)
    
def binomialCoefficients(n, k):
    nUnderK = factorial(n) / (factorial(k) * factorial(n - k))
    return nUnderK
    
#Now for the actual funtion:
    
def forwardsDifferentiationKTimesEfficient(xn, yn, k):
    dydx = []
    
    for n in range(len(xn)-k):
        y_ = 0
        for i in range(k+1):
            y_ += (-1)**i * binomialCoefficients(k, i) * yn[n+k-i] 
        dydx.append(y_ / (xn[1] - xn[0])**k)
    return dydx

#BACKWARDS NUMERICAL DIFFERENTATION

def backwardsDifferentiation(xn,yn):
    
    dydx = []
    
    for i in range(1,len(xn)):
        dydx.append((yn[i] - yn[i-1]) / (xn[i] - xn[i-1]))
        
    return dydx

#BACKWARDS NUMERICAL DIFFERENTATION K TIMES (THIS USES THE FUNCTION ABOVE) NOTE: THIS IS VERY EXPENSIVE COMPUTATIONALLY BUT WILL WORK, THE BETTER VERSION BELOW WILL USE THE BINOMIAL SERIES TO QUICKEN THINGS IF THE STEP SIZE IS EQUAL EVERYWHERE

def backwardsDifferentiationKTimes(xn,yn, k):
    dydx = forwardsDifferentiation(xn,yn)
    for i in range(1, k):
        dydx = backwardsDifferentiation(xn[i:], dydx) #The x-s have to be split since we lose a node with each order of differentiation
    return dydx

#FORWARDS NUMERICAL DIFFERENTATION K TIMES EFFICIENT WAY WITH EQUIDISTANT NODES. NOTE: this uses the factorial and binominalCoefficient functions as well
    
def backwardsDifferentiationKTimesEfficient(xn, yn, k):
    dydx = []
    
    for n in range(k, len(xn)):
        y_ = 0
        for i in range(k+1):
            y_ += (-1)**i * binomialCoefficients(k, i) * yn[n-i] 
        dydx.append(y_ / (xn[1] - xn[0])**k)
    return dydx

#RIGHT RIEMANN SUM

def rightRiemannSum(x,y):
    #Finding the step sizes between x-s
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    #Calculating the right Riemann sum:
    rRSum = 0
    for i in range(len(x)-1):
        rRSum += y[i+1] * h[i]
        
    return rRSum

#LEFT RIEMANN SUM:
    
def leftRiemannSum(x,y):
    #Finding the step sizes between x-s
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    #Calculating the left Riemann sum:
    lRSum = 0
    for i in range(len(x)-1):
        lRSum += y[i] * h[i]
        
    return lRSum

#UPPER RIEMANN SUM
    
def upperRiemannSum(x,y):
    #Finding the step sizes between x-s
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    #Calculating the upper Riemann sum:
    upperRSum = 0
    for i in range(len(x)-1):
        rRSum = y[i+1] * h[i]
        lRSum = y[i] * h[i]
        if rRSum > lRSum:
            upperRSum += rRSum
        else:
            upperRSum += lRSum
    return upperRSum

#LOWER RIEMANN SUM  
    
def lowerRiemannSum(x,y):
    #Finding the step sizes between x-s
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    #Calculating the upper Riemann sum:
    upperRSum = 0
    for i in range(len(x)-1):
        rRSum = y[i+1] * h[i]
        lRSum = y[i] * h[i]
        if rRSum < lRSum:
            upperRSum += rRSum
        else:
            upperRSum += lRSum
    return upperRSum



#TRAPEZIUM RULE FOR EQUIDISTANT NODES

def trapeziumRuleWithEquidistantNodes(x,y):
    
    #Finding the number of nodes (n) and the subinrerval (h) (here assumed to be the same everywhere)
    n = len(x)
    h = x[1] - x[0]
    
    #Adding the first and last datapoints 
    integral = h * (y[0] / 2 + y[-1]/2)
    
    #Adding all other datapoints, (the sum part of the formula)
    for i in range(1,n-1):
        integral += h * y[i] #IMPORTANT NOTE: Here I multiply each value by h and thus open the bracket this is different from doing not openening the bracket and multiplying by h at the very end
        
    return integral

#TRAPEZIUM RULE FOR NON-EQUIDISTANT NODES

def trapeziumRuleGeneral(x,y):
    
    #Finding the number of nodes (n) and the subinrerval (h) between each neighbouring nodes (Here the nodes might not be equidistant!!)
    n = len(x)
    h = []
    for i in range(n-1):
        h.append(x[i+1] - x[i]) 
        
    integral = 0
    
    for i in range(n-1):
        
        integral += ((y[i+1]+y[i]) * h[i]) / 2
        
    return integral

#TRAPEZIUM RULE FOR NON-EQUIDISTANT NODES, DOUBLE INTEGRAL:

def trapeziumRuleDoubleIntegral(x_0, x_n, stepSizeXY):
    # set the x range, not including the boundaries
    x = np.arange(x_0 + stepSizeXY, x_n, stepSizeXY)
    N = len(x)
    # the y range depends of the various values of x, and cannot be fixed here
    
    # integrate in dy, for all the value of x, i.e. find G(x)
    
    G = np.zeros(N)
    # for every x
    for i in range(0,N):
        # determine the boundaries m and p for this x, NOTE: here I calculate these to be where the function slices the 
        mx = 423 # CHANGE THIS TO THE NUMBER NEEDED OR IF YOU WANT VOLUME UNDER A DOME JUST THE FUNCTION z(x,y) with y =0 !!!
        px = 423 # CHANGE THIS TO THE NUMBER NEEDED OR IF YOU WANT VOLUME UNDER A DOME JUST THE FUNCTION z(x,y) with y =0 !!!
        # set the y points for this x, not including the boundaries
        y = np.arange(-mx+stepSizeXY,px,stepSizeXY)
        z = np.zeros(len(y))
        # determine the values of the function z(x,y)
        for j in range(0,len(y)):
             z[j] = np.sqrt(25-x[i]**2-y[j]**2) # CHANGE THIS TO THE FUNCTION z(x,y) !!!!
        
        # integrate in dy from cx to dx (for this specific x)
        G[i] = trapeziumRuleGeneral(y,z) # G(x)
    
    # integrate G(x) in dx
    I = trapeziumRuleGeneral(x,G)
    return I

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



 #____WORKING WITH IMAGES____

#Reading in the image into a 3 layered matrix:

def readImage(fileName):
    
    Picture = pl.imread(fileName)
    
    return Picture  #Remember that the JPG file is just 3 layers of a matrix, each layer represents the RGB conde for the pixel it represents

#Displaying a 3 layer matrix as an image:

def showImage(Picture):
    
    pl.imshow(Picture)
    pl.show() #Make a new kernel for the next thing to display 
    
#Saving an image (3 layered matrix) to a file:
    
def saveImage(fileName, Picture):
    
    pl.imsave(fileName, Picture)
    
def compressingAnImage(Picture, CompressionRatio):
    smallPicture = np.zeros((int(Picture.shape[1] / CompressionRatio),int(Picture.shape[0] / CompressionRatio),3)) #Making the new target 3 layered array

    smallPicture = Picture[0: Picture.shape[1] : CompressionRatio, 0:Picture.shape[0] : CompressionRatio, :] #Entering only every n-th pixel into the new small image

    pl.imshow(smallPicture) #Showing the new image
    pl.show() #Make a new kernel for the next thing to display 

    
#RESIZEING AN IMAGE WITH INTERPOLATION
    
    #Using lagrangian intepolation
    
def imageResizeWithLagrangian(smallPicture, resizeRatio):

    xn = np.ndarray(smallPicture.shape[1]) #Defining the points we know, xn 
    for i in range(smallPicture.shape[1]):
        xn[i] = i
    
    N = resizeRatio #Defining the size increase
    
    x = np.linspace(0, smallPicture.shape[1], smallPicture.shape[1] * N) #Defining the x range where we want interpolated values
    largeImage1 = np.ndarray((smallPicture.shape[0], smallPicture.shape[1] * N, 3)) #Defining the matrix for the new image (only re-sized in teh x direction)
    
    #Finding the values for the missing points
    
    for i in range(3): #There are three layers for RGB
        for j in range(smallPicture.shape[1]): #Going over each part of the matrix row by row
            yn = smallPicture[j,:,i]
            y = LagrangianInterpolation(xn,yn,0,0,x)
            largeImage1[j,:,i] = y
    
    pl.imshow(largeImage1.astype(int))
    pl.show()
    
    
    #Next, in the y direction:
    
    xn = np.ndarray(smallPicture.shape[0]) #Defining the points we know, xn 
    for i in range(smallPicture.shape[0]):
        xn[i] = i
    
    
    
    x = np.linspace(0, smallPicture.shape[0], smallPicture.shape[0] * N) #Defining the x range where we want interpolated values
    largeImage2 = np.ndarray((smallPicture.shape[0] * N, smallPicture.shape[1] * N, 3)) #Defining the matrix for the new image (only re-sized in teh x direction)
    
    #Finding the values for the missing points
    
    for i in range(3): #There are three layers for RGB
        for j in range(smallPicture.shape[1] * N): #Going over each part of the matrix row by row
            yn = largeImage1[:,j,i]
            y = LagrangianInterpolation(xn,yn,0,0,x)
            largeImage2[:,j,i] = y
            
            
    largeImage2 = np.trunc(largeImage2)
    largePicture = (largeImage2.astype(int))
    return largePicture
    
    #Using spline interpolation
    

#First in the x direction
def imageResizeWithSplines(smallPicture, resizeRatio):

    xn = np.ndarray(smallPicture.shape[1]) #Defining the points we know, xn 
    for i in range(smallPicture.shape[1]):
        xn[i] = i
    
    N = resizeRatio #Defining the size increase
    
    x = np.linspace(0, smallPicture.shape[1], smallPicture.shape[1] * N) #Defining the x range where we want interpolated values
    largeImage1 = np.ndarray((smallPicture.shape[0], smallPicture.shape[1] * N, 3)) #Defining the matrix for the new image (only re-sized in teh x direction)
    
    #Finding the values for the missing points
    
    for i in range(3): #There are three layers for RGB
        for j in range(smallPicture.shape[1]): #Going over each part of the matrix row by row
            yn = smallPicture[j,:,i]
            y = splines(xn,yn,0,0,x)
            largeImage1[j,:,i] = y
    
    pl.imshow(largeImage1.astype(int))
    pl.show()
    
    
    #Next, in the y direction:
    
    xn = np.ndarray(smallPicture.shape[0]) #Defining the points we know, xn 
    for i in range(smallPicture.shape[0]):
        xn[i] = i
    
    
    
    x = np.linspace(0, smallPicture.shape[0], smallPicture.shape[0] * N) #Defining the x range where we want interpolated values
    largeImage2 = np.ndarray((smallPicture.shape[0] * N, smallPicture.shape[1] * N, 3)) #Defining the matrix for the new image (only re-sized in teh x direction)
    
    #Finding the values for the missing points
    
    for i in range(3): #There are three layers for RGB
        for j in range(smallPicture.shape[1] * N): #Going over each part of the matrix row by row
            yn = largeImage1[:,j,i]
            y = splines(xn,yn,0,0,x)
            largeImage2[:,j,i] = y
            
            
    largeImage2 = np.trunc(largeImage2)
    largePicture = (largeImage2.astype(int))
    return largePicture

    

 #____INTERPOLATION WITH UNSTRUCTURED GRIDS____

#Testing if 4 points are co-planar
    
def CoPlanarTest (r1,r2,r3,r4): #Enter the 4 points as position vectors

    A = [[r4[0]-r1[0],r4[1]-r1[1], r4[2]-r1[2]],[r4[0]-r2[0],r4[1]-r2[1], r4[2]-r2[2]],[r4[0]-r3[0],r4[1]-r3[1], r4[2]-r3[2]]]
    print(A)
    b = np.linalg.det(A)
    print(b)
    if b != 0:
         print("The fourth point is not co-planar with the 3 known points")
    else:
        print("You good, the fourth is also coplanar")
        
        
#Interpolation over a triangle with the nearest neghbour method:


def triangleInterponaltionNearestNeighbour(r1,r2,r3,f1,f2,f3,r4): #r1, r2, r3 are the position vectors of the 3 nodes that define the triangle. f1, f2, f3 are the values the function takes at that point. r4 is the position vector of the point of interest. The result will be the interponlated value f4 at r4.
    
    #Calculating the weights for each direction based on the distance between the point of interest and the points known
    w = []
    w.append(1/(np.sqrt((r1[0] - r4[0]) ** 2 + (r1[1] - r4[1]) ** 2)))
    w.append(1/(np.sqrt((r2[0] - r4[0]) ** 2 + (r2[1] - r4[1]) ** 2)))    
    w.append(1/(np.sqrt((r3[0] - r4[0]) ** 2 + (r3[1] - r4[1]) ** 2)))
    
    #Caclulating the weighted average at the point of interest
    f4 = (w[0] * f1 + w[1] * f2 + w[2] * f3) / (w[0] + w[1] + w[2])
    
    return f4

#Interpolation over a triangle with the Barycentric coordinates method:

def triangleInterponaltionBarycentric (r1,r2,r3,f1,f2,f3,r4):#r1, r2, r3 are the position vectors of the 3 nodes that define the triangle. f1, f2, f3 are the values the function takes at that point. r4 is the position vector of the point of interest. The result will be the interponlated value f4 at r4.
    
    #Instead of doing the matrix manipulations, I just put in the algebraic result to find the lambda values:
    lamda1 = (((r2[1] - r3[1]) * (r4[0] -r3[0]) + (r3[0]-r2[0]) * (r4[1] - r3[1])) / (((r2[1] - r3[1]) * (r1[0] - r3[0])) + (r3[0] - r2[0]) * (r1[1] - r3[1])))
    lamda2 = (((r3[1] - r1[1]) * (r4[0] -r3[0]) + (r1[0]-r3[0]) * (r4[1] - r3[1])) / (((r2[1] - r3[1]) * (r1[0] - r3[0])) + (r3[0] - r2[0]) * (r1[1] - r3[1])))
    lamda3 = 1 - lamda1 - lamda2
    
    #Caclulating the weighted average at the point of interest
    f4 = lamda1 * f1 + lamda2 * f2 + lamda3 * f3
    
    return f4