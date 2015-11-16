'''
Created on Nov 11, 2015

@author: micheleceru
'''
from sklearn.datasets import load_iris
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

import cProfile


def getNewClusterForPoint(point,clusters): 
    listOfCLusters=[]
    for clt in clusters:
        listOfCLusters.append(clt.centroid)
    return clusters[spatial.KDTree(np.array(listOfCLusters)).query(point)[1]] 

def mykmeans(dataPoints,maxClusterNumbers,maxIter=100):   
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(len(dataPoints))
    dataPoints = dataPoints[permutation] 
    data = generateStartingCentroid(dataPoints,maxClusterNumbers,kmeansTypy="plus")
    centroids=data[0:maxClusterNumbers]
    
    clusters = [Cluster(point,[point]) for point in centroids]    
    for nIter in range(maxIter):
    #To exclude the three centroids that were randomly selected from the data    
        allThePoints = data[maxClusterNumbers:] if nIter==0 else data 
        for cl in clusters:   
            cl.pointInCluster=[]# Erase all the point in the cluster before starting with the new iteration
        for point in allThePoints:
            newCluster=getNewClusterForPoint(point,clusters)
            newCluster.pointInCluster.append(point) 
        for cl in clusters:
            #If in the cluster there is at list one point:
            if len(cl.pointInCluster)!=0:
                #New centroid placed on the average of the points in the cluster
                newCentroid=np.mean(np.array(cl.pointInCluster),axis=0)
                cl.centroid=newCentroid
        
    #for cl in clusters:
    #    print "cluster: ", cl.centroid,"contains: ",len(cl.pointInCluster),"points"
    #plotClusters(clusters) 
    
class Cluster:
    centroid=[]
    pointInCluster=[]
    def __init__(self,centroid,pointInCluster):
        self.centroid=centroid
        self.pointInCluster=pointInCluster
    def getDistanceFromCentroid(self,point):
        return np.linalg.norm(self.centroid-point)

def plotClusters(clusters):
    for i in range(len(clusters)):
        iCluster= np.array(clusters[i].pointInCluster)
        plt.scatter(iCluster[:,0], iCluster[:,1],c=np.random.rand(3,1))
        centroid=clusters[i].centroid
        plt.scatter(centroid[0], centroid[1],c="gold")
    #plt.show()
    

#Given a list of centroids and a point, return the closer centroid from this point 
def getCloserCentroid(point,centroids): 
    return centroids[spatial.KDTree(np.array(centroids)).query(point)[1]] 

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
                distList.append(dist)
            weightProb=distList/sum(distList)
            indexes=range(len(dataPoints[i+1:]))
            index= np.random.choice(indexes, p=weightProb)
            #Add 1 because the point was randomly chosen on the n-1 point range so the index is one before the real one
            centroids.append(dataPoints[index])
            #Put the new centroid at the begining of the array
            dataPoints = np.concatenate((dataPoints[index:index+1],dataPoints[:index],dataPoints[index+1:]),axis=0)
        #print "starting centroid:",centroids
        return dataPoints

def start():
    iris = load_iris()
    data=iris.data
    mykmeans(data,maxClusterNumbers=3)     

if __name__ == '__main__':
    cProfile.run('start()')
