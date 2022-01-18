#____NUMERICAL DIFFERENTIATION____

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



