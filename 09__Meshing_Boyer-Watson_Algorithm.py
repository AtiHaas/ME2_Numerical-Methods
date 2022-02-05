#Importing neccesary libraries

import matplotlib.pyplot as pl
import numpy as np
import random


#___DEFINING ALL THE CLASSES WE WILL BE NEEDING__ (with printing and plotting functions for testing)

#Nodes, where measuremets are taken:

class Nodes:
    
    def __init__(self, x, y):
        self.xCoordinate = x
        self.yCoordinate = y
    
    def __str__(self):
        toPrint = " X Cooridnate = " + str(self.xCoordinate) +  "\n y Cooridnate = " + str(self.yCoordinate)
        return toPrint
    
    def __eq__(self, other):
        return self.xCoordinate == other.xCoordinate and self.yCoordinate == other.yCoordinate
    
    def plot(self, color = None):
        pl.scatter(x,y, c = color)

        
#Triangles made up of Nodes:
        
class Triangle:
    
    def __init__(self, point1, point2, point3):
        self.NodeA = [point1.xCoordinate, point1.yCoordinate]
        self.NodeB = [point2.xCoordinate, point2.yCoordinate]
        self.NodeC = [point3.xCoordinate, point3.yCoordinate]
        self.Edgea = [self.NodeB, self.NodeC]
        self.Edgeb = [self.NodeA, self.NodeC]
        self.Edgec = [self.NodeA, self.NodeB]
        
    def __str__(self):
        toPrint = " Node A is: " + str(self.NodeA) + "\n Node B is: " + str(self.NodeB) + "\n Node C is: " + str(self.NodeC) + "\n ____ "
        return toPrint
    
    def __eq__(self, other):
        if ((self.NodeA == (other.NodeA or other.NodeB or other.NodeC)) and (self.NodeB == (other.NodeA or other.NodeB or other.NodeC)) and (self.NodeC == (other.NodeA or other.NodeB or other.NodeC))):
            equal = True
        else:
            equal = False
        return equal
    
    #Checking if a given node is in the circumsircle of the triangle:
    def isPointInCircle(self, point):
        #See: https://en.wikipedia.org/wiki/Delaunay_triangulation
        test_ = [
            [self.NodeA[0] - point.xCoordinate, self.NodeA[1] - point.yCoordinate, ((self.NodeA[0])**2 - (point.xCoordinate)**2) + ((self.NodeA[1])**2 - (point.yCoordinate)**2)],
            [self.NodeB[0] - point.xCoordinate, self.NodeB[1] - point.yCoordinate, ((self.NodeB[0])**2 - (point.xCoordinate)**2) + ((self.NodeB[1])**2 - (point.yCoordinate)**2)],
            [self.NodeC[0] - point.xCoordinate, self.NodeC[1] - point.yCoordinate, ((self.NodeC[0])**2 - (point.xCoordinate)**2) + ((self.NodeC[1])**2 - (point.yCoordinate)**2)]]
    
        test = np.linalg.det(test_)
    
        if (test > 0):
            inside = True
        else:
            inside = False
        
        return inside
    
    #Checking if the triangle shares an edge with another triangle:
    
    def sharesEdgeWith(self, other): #Takes two triangles and tells us if they shar an edge or not
        test_ = (((self.NodeA == other.NodeA) or (self.NodeA == other.NodeB) or (self.NodeA == other.NodeC))
                  and (self.NodeB == other.NodeA) or (self.NodeB == other.NodeB) or (self.NodeB == other.NodeC)
                 or
                 ((self.NodeB == other.NodeA) or (self.NodeB == other.NodeB) or (self.NodeB == other.NodeC))
                  and (self.NodeC == other.NodeA) or (self.NodeC == other.NodeB) or (self.NodeC == other.NodeC)

                 or 
                 ((self.NodeA == other.NodeA) or (self.NodeA == other.NodeB) or (self.NodeA == other.NodeC))
                  and (self.NodeC == other.NodeA) or (self.NodeC == other.NodeB) or (self.NodeC == other.NodeC))
    
            
        if (test_ == True):
            shares = True
        else:
            shares = False
            
        return shares
      
    #Checking if the triangle has a given edge:
          
    def sharesEdge(self, edge):
        test_ = (( ((edge[0] == self.Edgea[0]) and (edge[1] == self.Edgea[1])) or ((edge[0] == self.Edgeb[0]) and (edge[1] == self.Edgeb[1])) or ((edge[0] == self.Edgec[0]) and (edge[1] == self.Edgec[1])) )
                 or
                 ( ((edge[1] == self.Edgea[0]) and (edge[0] == self.Edgea[1])) or ((edge[1] == self.Edgeb[0]) and (edge[0] == self.Edgeb[1])) or ((edge[1] == self.Edgec[0]) and (edge[0] == self.Edgec[1])) )
                 )
    
        if (test_ == True):
            shares = True
        else:
            shares = False
            
        return shares
    
    #Plotting the triangle:
    
    def plot(self):
        toPlotX = [self.NodeA[0], self.NodeB[0], self.NodeC[0]]
        toPlotY = [self.NodeA[1], self.NodeB[1], self.NodeC[1]]
        pl.plot(toPlotX, toPlotY, c = "red")
        pl.scatter(toPlotX,toPlotY, c = "red")
        toPlotX = [self.NodeA[0],  self.NodeC[0]]
        toPlotY = [self.NodeA[1],  self.NodeC[1]]
        pl.plot(toPlotX, toPlotY, c = "red")
        pl.scatter(toPlotX,toPlotY, c = "red")
    
    
    
class Triangulation:
    
    def __init__ (self):
        self.listOfTriangles = []
        
    def __str__(self):
        toPrint = "The triangles in this particular triangulation are:"
        for i in range(len(self.listOfTriangles)):
            toPrint += f"\n Triangle {i+1} is:"
            toPrint += f"\n {self.listOfTriangles[i].NodeA}"
            toPrint += f"\n {self.listOfTriangles[i].NodeB}"
            toPrint += f"\n {self.listOfTriangles[i].NodeC}"
            toPrint += f"\n _____"
        return toPrint
    
    #Adding a triangle to the triangulation:
    
    def addTriangle(self, triangleToAdd):
        self.listOfTriangles.append(triangleToAdd)
        
    #Removeing a triangle to the triangulation:
        
    def removeTriangle(self, triangleToRemove):
        self.listOfTriangles.remove(triangleToRemove)
        
        
    #Plotting the triangulation
    def plotTriangulation(self):
        
        for i in range(len(self.listOfTriangles)):
            toPlotX = []
            toPlotY = []
            for j in range(1):
                toPlotX = [self.listOfTriangles[i].NodeA[0], self.listOfTriangles[i].NodeB[0], self.listOfTriangles[i].NodeC[0]]
                toPlotY = [self.listOfTriangles[i].NodeA[1], self.listOfTriangles[i].NodeB[1], self.listOfTriangles[i].NodeC[1]]
                pl.plot(toPlotX, toPlotY, c = "black")
                pl.scatter(toPlotX,toPlotY, c = "black")
                toPlotX = [self.listOfTriangles[i].NodeA[0],  self.listOfTriangles[i].NodeC[0]]
                toPlotY = [self.listOfTriangles[i].NodeA[1],  self.listOfTriangles[i].NodeC[1]]
                pl.plot(toPlotX, toPlotY, c = "black")
                pl.scatter(toPlotX,toPlotY, c = "black")
        pl.show()

#___DEFINING ALL THE FUNCTIONS WE WILL BE NEEDING__
    
    
#Generating a random number between a and b, to help with testing:
def RandomBetween (a, b):
    Rand = random.random()*(b-a) + a
    return Rand     
            
#Making a list of triangles unique (This is really slow but i cant order this so this is the best idea i have)

def makeUnique(listOfTriangles):
    toRemove = []
    for i in range(len(listOfTriangles)):
        search_ = listOfTriangles[i]
        for j in range(i, len(listOfTriangles)):
            if (search_ == listOfTriangles[j]):
                toRemove.append(listOfTriangles[j])
    
    for i in range(len(toRemove)):
        listOfTriangles.remove(toRemove[i])
    
    return listOfTriangles
                

#Given the set of points, in two lists of x and y coordinates, making them into a list of Nodes as defined above:       

def makePointList(x,y):
    #Creating the list that will hold the points
    pointList = []
    for i in range(len(x)):
        Node_ = Nodes(x[i], y[i])
        pointList.append(Node_)
    
    return pointList

#Create Supertriangle (The triangle that will definately contain all points given)
    
def makeSuperTriangle(x,y):
    #See https://web.mit.edu/alexmv/Public/6.850-lectures/lecture09.pdf
    
    M = max([max(x), max(y)])
    superA = Nodes(3 * M, 0)
    superB = Nodes(0, 3 * M)
    superC = Nodes(-3 * M, -3 * M)
    superTriangle = Triangle(superA, superB, superC)
    
    return superTriangle

#The Boyer-Watson algorthm for making the optimal triangulation: (https://www.youtube.com/watch?v=GctAunEuHt4)

def delunayTriangulation(x,y): 
    
    #Making the cooordinates into points
    pointList = makePointList(x,y)
    
    
    #Creating the supertriangle:
    superTriangle = makeSuperTriangle(x,y)
   
    
    #Creating the triangulation and adding the super triangle
    delunay = Triangulation()
    delunay.addTriangle(superTriangle)

    
    #Going over adding each point one by one and making a new mesh:
    
    for i in range(len(pointList)):
        
        #Creating a list of bad triangles
        badTriangles = Triangulation()
        
        #Finding the bad triangles:
        for j in range(len(delunay.listOfTriangles)):
            if (delunay.listOfTriangles[j].isPointInCircle(pointList[i])):
                badTriangles.addTriangle(delunay.listOfTriangles[j])
    
 
        
        #Adding the new edges
        for k in range(len(badTriangles.listOfTriangles)):
            
            #First seeing if the given bad triangle shares an edge with other bad traingles
            badTraingle_ = badTriangles.listOfTriangles[k]
            sharesA = 0
            sharesB = 0
            sharesC = 0
            
            for l in range(len(badTriangles.listOfTriangles)):
                if (badTriangles.listOfTriangles[l].sharesEdge(badTraingle_.Edgea)):
                    sharesA += 1
                if (badTriangles.listOfTriangles[l].sharesEdge(badTraingle_.Edgeb)):
                    sharesB += 1
                if (badTriangles.listOfTriangles[l].sharesEdge(badTraingle_.Edgec)):
                    sharesC += 1
            
            #The 3 new triangles that I might want to add (I only want to add the ones that are not shared by two bad triangles)
            newTriangle1 = Triangle(Nodes(badTriangles.listOfTriangles[k].NodeA[0], badTriangles.listOfTriangles[k].NodeA[1]), Nodes(badTriangles.listOfTriangles[k].NodeB[0], badTriangles.listOfTriangles[k].NodeB[1]), pointList[i])
            newTriangle2 = Triangle(Nodes(badTriangles.listOfTriangles[k].NodeB[0], badTriangles.listOfTriangles[k].NodeB[1]), Nodes(badTriangles.listOfTriangles[k].NodeC[0], badTriangles.listOfTriangles[k].NodeC[1]), pointList[i])
            newTriangle3 = Triangle(Nodes(badTriangles.listOfTriangles[k].NodeC[0], badTriangles.listOfTriangles[k].NodeC[1]), Nodes(badTriangles.listOfTriangles[k].NodeA[0], badTriangles.listOfTriangles[k].NodeA[1]), pointList[i])
            
            #Adding the right traingles to the triangulation:
            if (sharesC < 2): #Two since it will always share the edge with itself
                delunay.addTriangle(newTriangle1)
            #delunay.plotTriangulation()  #This just animates the proscedure
            
                
            if (sharesA < 2): #Two since it will always share the edge with itself
                delunay.addTriangle(newTriangle2)
            #delunay.plotTriangulation() #This just animates the proscedure
            
            if (sharesB < 2): #Two since it will always share the edge with itself
                delunay.addTriangle(newTriangle3)
            #delunay.plotTriangulation() #This just animates the proscedure
                
        #Delete the bad triangles:
        for m in range(len(badTriangles.listOfTriangles)):

            delunay.removeTriangle(badTriangles.listOfTriangles[m])
            #delunay.plotTriangulation() #This just animates the proscedure
              
    #Delete all triangles containing vertecies from the super triangle
       
    toRemove = []
    
    for n in range(len(delunay.listOfTriangles)):

        if ((delunay.listOfTriangles[n].NodeA == superTriangle.NodeA or delunay.listOfTriangles[n].NodeA == superTriangle.NodeB) or (delunay.listOfTriangles[n].NodeA == superTriangle.NodeC) or
            (delunay.listOfTriangles[n].NodeB == superTriangle.NodeA or delunay.listOfTriangles[n].NodeB == superTriangle.NodeB) or (delunay.listOfTriangles[n].NodeB == superTriangle.NodeC) or
            (delunay.listOfTriangles[n].NodeC == superTriangle.NodeA or delunay.listOfTriangles[n].NodeC == superTriangle.NodeB) or (delunay.listOfTriangles[n].NodeC == superTriangle.NodeC)
            ) :
                
            toRemove.append(delunay.listOfTriangles[n])
            
    for o in range(len(toRemove)):
        delunay.listOfTriangles.remove(toRemove[o])
        #delunay.plotTriangulation()  #This just animates the proscedure
    
    return delunay
    
    


#Testing the function:


#Generating a random set of n points:

n = 100

x = []
y = []

for i in range(n):
    x.append(RandomBetween(-12,16))
    y.append(RandomBetween(-16,19))


#Running and plotting the function:
    
delunayTriangulation1 = delunayTriangulation(x,y)
delunayTriangulation1.plotTriangulation()
