#___NUMERICAL INTEGRATION___

import numpy as np

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

#SIMPSONS RULE WITH UNIFORMLY DISTRIBUTED POINTS

def simpsonsRule(xn,yn):
    h = xn[1] - xn[0] #First, finding the step size
    I = h / 3 * (yn[0] + yn[-1]) #Creating the integral and adding the last and first node
    
    for i in range(1,len(xn)-1):
        if i % 2 == 1:
            I += h / 3 * 4 * yn[i]
        else:
            I += h / 3 * 2 * yn[i] #Note that I multilpy with h/3 inside and that changes the end a tiny bit
    return I

#ADAPTIVE SIMPSONS RULE, NOTE THAT HERE THE FUNCTION WILL HAVE TO BE KNOWN 
    
def adaptiveSimpsonsRule(domainLowerBound, domainUpperBound, tolerance):
    numberOfSubintervals = 1 #Start with one
    xn = np.linspace(domainLowerBound, domainUpperBound, numberOfSubintervals+1)
    yn = np.sin(xn) #RE-WRITE THE y = f(x) FUNCTION HERE
    
    I = simpsonsRule(xn,yn)
    
    error = tolerance * 10 #Set an artificially large error to enter the while loop:
    
    while error >= tolerance:
        numberOfSubintervals *= 2
        xn = np.linspace(domainLowerBound, domainUpperBound, numberOfSubintervals+1)
        yn = np.sin(xn) #RE-WRITE THE y = f(x) FUNCTION HERE
        I_ = simpsonsRule(xn,yn)
        error = np.abs(1 / 15 * (I-I_))
        I = I_
    
    return I
