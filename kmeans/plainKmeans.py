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
    mykmeans(data,maxClusternumbers=3)

def mykmeans(dataPoints,maxClusternumbers,maxIter=100):    
    random_seed = 10312003
    rng = np.random.RandomState(random_seed)
    permutation = rng.permutation(len(dataPoints))
    data=dataPoints[permutation]
    centroids=data[0:maxClusternumbers]
    clusters = [Cluster(point,[point]) for point in centroids]
    
    for nIter in range(maxIter):
    #Te exclude the trhee centroide that were randomly selected from the data    
        allThePoints = data[maxClusternumbers:] if nIter==0 else data 
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
    for cl in clusters:
        print "cluster: ", cl.centroid,"contains: ",len(cl.pointInCluster),"points"
    plotClusters(clusters) 
    
class Cluster:
    centroid=[]
    pointInCluster=[]
    
    def __init__(self,centroid,pointInCluster):
        self.centroid=centroid
        self.pointInCluster=pointInCluster
    
    def getDistanceFromCentroid(self,point):
        return np.linalg.norm(self.centroid-point)


def plotClusters(clusters):
    #colors = "bgrcmykwbgrcmykw"
    #print len(clusters)
    for i in range(len(clusters)):
        iCluster= np.array(clusters[i].pointInCluster)
        plt.scatter(iCluster[:,1], iCluster[:,2],c=np.random.rand(3,1), label="Class 3")
    plt.show()


if __name__ == '__main__':
    start()