#FOURIER TRANSFORM AND FILTERING

import numpy as np
import matplotlib.pyplot as pl

#DISCRTETE SLOW FOURIER TRANSFORM


def fourierTransform(x,y): #NOTE: for this to work x will have to be evenly spaced
    #Defining the array that will hold the frequency domain
    F = []

    
    #Finding the sampling frequency:
    h = x[2] - x[1]
   
    
    #Calculating each value of of the transform in the frequency domain:
    for k in range(len(x)):
        A_ = 0 #Real part
        B_ = 0 #Imaginary part
        for i in range(len(x)):
            A_ += y[i] * np.cos(-(2 * np.pi * k * i) / len(x))
            B_ += y[i] * np.sin(-(2 * np.pi * k * i) / len(x))
        F.append((np.sqrt(A_**2 + B_**2)))
        
    #Calculating the frequencies (so basically the x coordinate of the fourier trasform plot)
    f = np.linspace(0, 1/(2*h), int(len(x)/2))
    
    #Returning the end result:
    return f, F


#SIGNAL FILTERING:

def filterSignal(x,y, minFrequency, maxFrequency): #Filter will take out the frequencies between minFrequency and maxFrequency
    
    #First finding the fourier transform:
    f, F = fourierTransform(x,y)
    
    #Finding the average of F (this will help us to find the spikes since those have to be above the average of the whole domain)
    F_average = 0
    for i in range(len(F)//2):
        F_average += F[i]
    
    F_average = F_average/(len(F)//2)
    
    #Finging the local maxima, so the values of the frequesncies that make it up
    
    w = [] #Defining the list that will hold the frequencies that build up the 
    
    for i in range(len(F)//2):
        if (F[i] > F_average and F[i-1] < F[i] > F[i+1]):
            w.append(f[i])
            
    #Rebuilding a new signal, with only the desired frequency:
    x2 = x
    y2 = np.linspace(0,0,len(x))
    
    for i in range(len(w)):
        if minFrequency < w[i] < maxFrequency:
            for j in range(len(y2)):
                y2[j] += np.sin(w[i] * 2 * np.pi * x2[j])
    
    return x2, y2

#Testing the function:
    
x = np.arange(0, 5.01, 0.01)
y = []

for i in x:
    y_ = np.sin(3 *2*np.pi* i) + np.sin(4 *2*np.pi * i) + np.sin(30 * 2 * np.pi * i)
    y.append(y_)
    
f,F = fourierTransform(x,y)

x2,y2 = filterSignal(x, y,1,10)

f2, F2 = fourierTransform(x2,y2)


#Plotting the answers:

pl.plot(x,y, c = "blue")
pl.plot(x2,y2, c = "red")
pl.legend(['Unfiltered Signal', 'Filtered Signal'])
pl.show()
pl.plot(f,F[:len(F)//2], c = "blue")
pl.plot(f2, F2[:len(F)//2], c = "red")
pl.legend(['Unfiltered Signal Frequencies', 'Filtered Signal Frequencies'])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
