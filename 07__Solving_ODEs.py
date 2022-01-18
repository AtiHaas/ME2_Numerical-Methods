#__SOLVING ORDINARY DIFFERENTIAL EQUATIONS___

import numpy as np

#DEFINING THE FUNCTION dy/dx = f(x,y)

def f(x, y): #Change dy/dx here
    a = y * np.exp(-x**2)
    return a


#FORWARDS EULER METHOD FOR SOLVING ODE-S

def forwardEuler(y_0, h, x_n): #Where y_0 is the initial condition, h is the stepsize, and x_n is the last point to which we want to find the solution
    xn = np.arange(0, x_n + h, h) #If the initial condition don't start in x = 0 then rewrite this line
    yn = [y_0] #Creating yn and adding the first node that we know
    
    for i in range(len(xn)-1):
        y_ = yn[i] + h * f(xn[i],yn[i]) 
        yn.append(y_)
    
    return xn, yn

#HEUNS METHOD FOR SOLVING ODE-S
    
def heunsMethod(y_0, h, x_n): #Where y_0 is the initial condition, h is the stepsize, and x_n is the last point to which we want to find the solution
    xn = np.arange(0, x_n + h, h) #If the initial condition don't start in x = 0 then rewrite this line
    yn = [y_0] #Creating yn and adding the first node that we know
    
    for i in range(len(xn)-1):
        k1 = h * f(xn[i],yn[i]) 
        k2 = h * f(xn[i+1], yn[i] + k1)
        y_ = yn[i] + 0.5 * (k1 + k2)
        yn.append(y_)
    
    return xn, yn 

#RK4 METHOD

def RK4(y_0, h, x_n):
    xn = np.arange(0, x_n + h, h) #If the initial condition don't start in x = 0 then rewrite this line
    yn = [y_0] #Creating yn and adding the first node that we know
    
    for i in range(len(xn)-1):
        k1 = h * f(xn[i], yn[i]) 
        k2 = h * f(xn[i] + h/2, yn[i] + k1/2)
        k3 = h * f(xn[i] + h/2, yn[i] + k2/2)
        k4 = h * f(xn[i] + h, yn [i] + k3)
        y_ = yn[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        yn.append(y_)
    
    return xn, yn


#ADAM-BASHFORD 4TH ORDER METHOD:

def adamBashford4thOrder(y_0, h, x_n):
    xn = np.arange(0, x_n + h, h) #If the initial condition don't start in x = 0 then rewrite this line
    yn = [y_0] #Creating yn and adding the first node that we know
    
    #Making sure that the step size is large enough for us to be able to do Adam-Bashford:
    
    if len(xn) < 3:
        print("Enter a smaller step size, Adam-Basford needs at least nodes to work dummy")
        return None
    
    #First we have to get the first 4 points with the RK4 predictor method in order to use the Adam Bashford method:
    
    for i in range(3):   
         k1 = h * f(xn[i], yn[i]) 
         k2 = h * f(xn[i] + h/2, yn[i] + k1/2)
         k3 = h * f(xn[i] + h/2, yn[i] + k2/2)
         k4 = h * f(xn[i] + h, yn [i] + k3)
         y_ = yn[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
         yn.append(y_)

    for i in range(3, len(xn)-1):
        y_ = yn [i] + h /24 * (55 * f(xn[i], yn[i]) - 59 * f(xn[i-1], yn[i-1]) + 37 * f(xn[i-2], yn[i-2]) - 9 * f(xn[i-3], yn[i-3]))
        yn.append(y_)
    
    
    return xn, yn
