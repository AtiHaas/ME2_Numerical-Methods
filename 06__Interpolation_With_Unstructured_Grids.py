#____INTERPOLATION WITH UNSTRUCTURED GRIDS____

import numpy as np

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


