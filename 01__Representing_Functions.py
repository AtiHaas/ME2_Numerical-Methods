import numpy as np
import matplotlib.pyplot as pl

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
   

