'''
Created on Nov 11, 2015

@author: micheleceru
'''

from sklearn.datasets import load_iris
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def getNewClusterForPoint(point,clusters): 
    listOfCLusters=[]
    for clt in clusters:
        listOfCLusters.append(clt.centroid)
    return clusters[spatial.KDTree(np.array(listOfCLusters)).query(point)[1]] 

def start():
    iris = load_iris()
    data=iris.data
    mykmeansMulti(data,maxClusterNumbers=3,numberOfLoop=2)
    #Don't know why but for maxClusterNumbers>51 the program breaks
    # plt.scatter(pointInOneCluster[:,0],pointInOneCluster[:,1],c=clusterColor)    
    #I should find out why!!!
    #Anyway commenting the plot it works fine. For  maxClusterNumbers=150 every point has its own cluster and the 
    #distortion is 0
    
def mykmeansMulti(dataPoints,maxClusterNumbers,maxIter=100,numberOfLoop=1):   
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(len(dataPoints))
    dataPoints = dataPoints[permutation] 
    
    bestPerformingDistortion = np.zeros(maxIter)
    for i in range(numberOfLoop):
        data=np.roll(dataPoints, i+maxClusterNumbers, axis=0)    
        data = generateStartingCentroid(data,maxClusterNumbers,kmeansTypy="plus")
        centroids=data[0:maxClusterNumbers]
        clusters = [Cluster(point,[point]) for point in centroids]
        
        distortionInIterations = np.zeros(maxIter)
        clustersInIterations = []
        for nIter in range(maxIter):
        #To exclude the three centroids that were randomly selected from the data    
            allThePoints = data[maxClusterNumbers:] if nIter==0 else data 
            for cl in clusters:   
                cl.pointInCluster=[]# Erase all the point in the cluster before starting with the new iteration
            for point in allThePoints:
                newCluster=getNewClusterForPoint(point,clusters)
                newCluster.pointInCluster.append(point) 
                
            storedInfoOfClusters = []
            for cl in clusters:         
                storedInfoOfClusters.append(Cluster(cl.centroid,cl.pointInCluster))
                #If in the cluster there is at list one point:
                if len(cl.pointInCluster)!=0:
                    #New centroid placed on the average of the points in the cluster
                    newCentroid=np.mean(np.array(cl.pointInCluster),axis=0)
                    cl.centroid=newCentroid
            #To plot the distortion curve for the best performing set of initial poinst:
            distortion = 0
            
            #Calculate distortion for each iteration
            for cl in clusters:   
                distortion = distortion + cl.computeClusterDistortion()
            distortionInIterations[nIter]=distortion 
            clustersInIterations.append(np.array(storedInfoOfClusters))
        #To pick the information of the best performing clusters:       
        if i==0:
            bestPerformingDistortion = distortionInIterations[:]
            bestPerformingClusters   = np.array(clustersInIterations)
        elif distortionInIterations[-1]<bestPerformingDistortion[-1]:
            bestPerformingDistortion = distortionInIterations[:]
            bestPerformingClusters   = np.array(clustersInIterations)
     
    ####To plot the best performance cluster with the trajectory of the centroid:                   
    for ncl in range(maxClusterNumbers):
        x=[]
        y=[]
        clusterColor=np.random.rand(3,1)
        for cl in bestPerformingClusters[:,ncl]:#Pick one centroid and look at the trajectory
            x.append(cl.centroid[0])
            y.append(cl.centroid[1])           
        plt.plot(x,y,c=clusterColor)
        #print "All the point in cluster:",ncl,": ", bestPerformingClusters[-1][ncl].pointInCluster  
        pointInOneCluster= np.array(bestPerformingClusters[-1][ncl].pointInCluster)
        plt.scatter(pointInOneCluster[:,0],pointInOneCluster[:,1],c=clusterColor)    
    plt.show()
    
    plt.clf()
    plt.scatter(range(maxIter),bestPerformingDistortion)
    plt.xlim(0.,100)
    plt.show()
    
    
        
        #Plot to decomment
        #plotClusters(clusters) 
        
    
class Cluster:
    centroid=[]
    pointInCluster=[]
    
    def __init__(self,centroid,pointInCluster):
        self.centroid=centroid
        self.pointInCluster=pointInCluster
    
    def getDistanceFromCentroid(self,point):
        return np.linalg.norm(self.centroid-point)
    
    def computeClusterDistortion(self):
        clusterDistortion=0
        for point in np.array(self.pointInCluster):
            clusterDistortion = clusterDistortion + np.linalg.norm(point-self.centroid)
        return  clusterDistortion
        #print np.array(self.pointInCluster)
        #dist = numpy.linalg.norm(-b)

def plotClusters(clusters):
    for i in range(len(clusters)):
        iCluster= np.array(clusters[i].pointInCluster)
        plt.scatter(iCluster[:,0], iCluster[:,1],c=np.random.rand(3,1))
        centroid=clusters[i].centroid
        plt.scatter(centroid[0], centroid[1],c="gold")
    plt.show()
    
def generateStartingCentroid(dataPoints,maxClusternumbers,kmeansTypy="plane"):
    if kmeansTypy == "plane":
        return dataPoints
    elif kmeansTypy == "plus":
        centroids = []
        centroids.append(dataPoints[0])
        for i in range(maxClusternumbers-1):
            distList=[]
            for point in dataPoints[i+1:]:
                dist = np.linalg.norm(point-getCloserCentroid(point,centroids))
                distList.append(dist**2)
            weightProb=distList/sum(distList)
            indexes=range(len(dataPoints[i+1:]))
            index= np.random.choice(indexes, p=weightProb)
            #Add 1 because the point was randomly chosen on the n-1 point range so the index is one before the real one
            centroids.append(dataPoints[index])
            #Put the new centroid at the begining of the array
            dataPoints = np.concatenate((dataPoints[index:index+1],dataPoints[:index],dataPoints[index+1:]),axis=0)
        print "starting centroid:",centroids
        return dataPoints
        
    else: #Here add the part for kmeans++
        print "plus"

#Given a list of centroids and a point, return the closer centroid from this point 
def getCloserCentroid(point,centroids): 
    return centroids[spatial.KDTree(np.array(centroids)).query(point)[1]] 

    
    #First centroid chosen randomly
    
        
   

if __name__ == '__main__':
    start()
