'''
Created on Nov 13, 2015

@author: micheleceru
'''

from sklearn.datasets import load_iris
import numpy as np
from scipy import spatial
#from sklearn.decomposition import PC

import matplotlib.pyplot as plt

def getNewClusterForPoint(point,clusters): 
    listOfCLusters=[]
    for clt in clusters:
        listOfCLusters.append(clt.centroid)
    return clusters[spatial.KDTree(np.array(listOfCLusters)).query(point)[1]] 
    
def start():
    iris = load_iris()
    data=iris.data
    mykmeans(data,maxClusternumbers=3,numberOfLoop=10)
    
def mykmeans(dataPoints,maxClusternumbers=3,maxIter=100,numberOfLoop=1):    
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(len(dataPoints))
    data=generateStartingCentroid(dataPoints,maxClusternumbers)

    bestPerformingDistortion = []
    bestPerformingCentroids = []
    for i in range(numberOfLoop):
        data=np.roll(data, numberOfLoop+3, axis=0)
        centroids=data[0:maxClusternumbers]
        clusters = [Cluster(point,[point]) for point in centroids]

        distortionInIterations = []
        centroidsInIterations = []
        for nIter in range(maxIter):
        #To exclude the trhee centroide that were randomly selected from the data    
            allThePoints = data[maxClusternumbers:] if nIter==1 else data 
            #print "Number of iteration: ", nIter

            for cl in clusters:   
                cl.pointInCluster=[]# Erase all the point in the cluster before starting with the new iteration

            for point in allThePoints:
                newCluster=getNewClusterForPoint(point,clusters)
                newCluster.pointInCluster.append(point)

            for cl in clusters:
                #If in the claster there is at list one point:
                if len(cl.pointInCluster)!=0:
                    #New centroid placed on the average of the points in the cluster
                    newCentroid=np.mean(np.array(cl.pointInCluster),axis=0)
                    cl.centroid=newCentroid
                
            #To plot the distortion curve for the best performing set of initial poinst:
            distortion=0
            centroids=[]
            for cl in clusters:   
                distortion = distortion + cl.computeClusterDistortion()
                centroids.append(cl.centroid)
            distortionInIterations.append(distortion) 
            centroidsInIterations.append(centroids)
        if i==0:
            bestPerformingDistortion = distortionInIterations[:]
            bestPerformingCentroids = centroidsInIterations[:]
        elif distortionInIterations[-1]<bestPerformingDistortion[-1]:
                bestPerformingDistortion = distortionInIterations[:]
                bestPerformingCentroids = centroidsInIterations[:]

        distortion = 0
        for cl in clusters:   
            distortion = distortion + cl.computeClusterDistortion()
        print distortion 
        
    #print "best performance: ",bestPerformingDistortion
    #print "best centroids: ", bestPerformingCentroids
    
    
    #plt.scatter(range(maxIter),bestPerformingDistortion)
    colors = "bgrcmykw"
    for i in range(maxClusternumbers):
        plt.plot(np.array(bestPerformingCentroids)[:,i][:,0], np.array(bestPerformingCentroids)[:,i][:,1], c=colors[i])
    plotClusters(clusters)
    plt.show()
    
def plotClusters(clusters):
    colors = "bgrcmykw"
    #print len(clusters)
    for i in range(len(clusters)):
        iCluster= np.array(clusters[i].pointInCluster)
        plt.scatter(iCluster[:,0], iCluster[:,1],c=colors[i], label="Class 3")
        
    
    
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

    
#Order  the dataPoints placing the centroid at the beginning of the numpy array
def generateStartingCentroid(dataPoints,maxClusternumbers):
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(len(dataPoints))
    dataPoints = dataPoints[permutation]
    centroids = []
    #First centroid chosen randomly
    centroids.append(dataPoints[0])
        
    for i in range(maxClusternumbers-1):
        distList=[]
        for point in dataPoints[i+1:]:
            dist = np.linalg.norm(point-getCloserCentroid(point,centroids))
            distList.append(dist)
        weightProb=distList/sum(distList)
        indexes=range(len(dataPoints[i+1:]))
        index= np.random.choice(indexes, p=weightProb)
        #Add 1 because the point was randomly chosen on the n-1 point range so the index is one before the real one
        centroids.append(dataPoints[index])
        #Put the new centroid at the begining of the array
        dataPoints = np.concatenate((dataPoints[index:index+1],dataPoints[:index],dataPoints[index+1:]),axis=0)
    print "starting centroid:",centroids
    return dataPoints

#Given a list of centroids and a point, return the closer centroid from this point 
def getCloserCentroid(point,centroids): 
    return centroids[spatial.KDTree(np.array(centroids)).query(point)[1]] 


if __name__ == '__main__':
    start()