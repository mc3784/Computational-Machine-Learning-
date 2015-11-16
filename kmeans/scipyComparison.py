'''
Created on Nov 15, 2015

@author: micheleceru
'''
from sklearn import cluster, datasets
import cProfile

def start():
    iris = datasets.load_iris()
    k_means = cluster.KMeans(3)
    k_means.fit(iris.data) 
#print k_means.labels_[::]
#print iris.target[::]


if __name__=="__main__":
    cProfile.run('start()')