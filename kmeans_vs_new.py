import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib

np.random.seed(10)

def takeInput(filename="bisecting.txt", delimiter="\t"):
    point_list = []
    with open(filename, "r") as f:
        line = f.readline()
        while((line.rstrip()) != ""):
            x, y = line.strip().split(delimiter)            
            x = float(x)
            y = float(y)
            line = f.readline()
            point_list.append((x, y))        
    return point_list

class Cluster:
    def __init__(self, point_list, centroid_tuple=(0,0)):
        self.color = np.random.uniform(5, 100)
        self.point_list = []
        self.point_list.extend(point_list)
        if (len(point_list)==0):
            self.centroid = centroid_tuple
        else:        
            self.centroid = self.updateCentroid()

    def updateCentroid(self):
        if (len(self.point_list)!=0):
            '''
            sumx = 0
            sumy = 0
            for tup in self.point_list:
                sumx += tup[0]
                sumy += tup[1]
            sumx /= len(self.point_list)
            sumy /= len(self.point_list)
            self.centroid = (sumx, sumy)
            '''
            self.centroid = np.average(self.point_list, axis=0)
        return self.centroid

    def SSE(self):
        sse = 0
        for tup in self.point_list:
            sse += (self.centroid[0] - tup[0])**2 + (self.centroid[1] - tup[1])**2 
        return sse
    
    def distance(self, point):
        return ((self.centroid[0] - point[0])**2 + (self.centroid[1] - point[1])**2)
    

def kmeans(point_list, k, centroids=[], SIMULATION = False):
    nplist = np.asarray(point_list , dtype='float') # [[1,2],[2,4],[3,6]]
    xlist = nplist[0: , 0]  # [1, 2, 3]
    ylist = nplist[0: , 1]  # [2, 4, 6]
    label = np.zeros(len(xlist), dtype='int32') # [0, 0, 0]

    #selecting k points as initial centroid
    maxx, maxy = np.max(nplist, axis = 0)
    minx, miny = np.min(nplist, axis = 0)

    if(len(centroids)!=k):   
        centroids = []
        for i in xrange(k):
            ranx = np.random.uniform(minx, maxx)
            rany = np.random.uniform(miny, maxy)
            centroids.append((ranx, rany))
        centroids =  np.asarray(centroids , dtype='float')

    ####plt.scatter(xlist, ylist)
    ####plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, color='r' )
    ####plt.show()
    
    # initial k clusters
    clusters = []
    cluster_colors =(cm.rainbow(np.linspace(0,1,k)))
    for i in xrange(k):
        c = Cluster([], centroids[i])
        c.color = cluster_colors[i]
        clusters.append(c) 

    # repeat
    # form k clusters by assigning all points to the closest centroid
    # recompute the centroid
    # until the centroids dnt change
    breakflag = False
    figiter = 0
    numpoints = len(point_list)
    while(True):
        if breakflag:
            break         
        for ip in xrange(numpoints):
            point = point_list[ip]
            dists = [0.0]*k        
            for i in xrange(k):
                dists[i] = (clusters[i].distance(point))
            min_dist_k = np.argmin(dists)
            label[ip] = min_dist_k
            clusters[min_dist_k].point_list.append(point)

        new_centroids = [(0,0)]*k    
        for i in xrange(k):        
            new_centroids[i] = (clusters[i].updateCentroid())

        new_centroids = np.asarray(new_centroids, dtype=float)
        if (np.allclose(centroids, new_centroids, .001, .001)):
            breakflag = True    
        centroids = new_centroids
        figiter = figiter + 1
        #print figiter

    if(SIMULATION):
        t=(cm.rainbow(np.linspace(0,1,len(xlist))))
        for i in xrange(len(xlist)):
            t[i] = clusters[label[i]].color
        plt.scatter(xlist,ylist, c=t)
        for i,cc in zip(range(k), cluster_colors):
            plt.plot(centroids[i][0],centroids[i][1], '+', mew=10, ms=25,c='black')
            plt.plot(centroids[i][0],centroids[i][1], '+', mew=5, ms=20,c=cc)
        plt.title("K-Means " + str(figiter))
        plt.show()
        
    return clusters

def bisectingKmeans(point_list, k, SIMULATION = False):
    output_clusters = []
    c = Cluster(point_list)
    list_clusters = [c]
    while(len(list_clusters) < k):
        # select a cluster from the list of cluster
        #picked_cluster_i = np.random.randint(0, len(list_clusters))
        sses = []
        for cl in list_clusters:
            sses.append(cl.SSE())
        picked_cluster_i = np.argmax(sses)    

        pick = list_clusters[picked_cluster_i]
        min_sum_sse = float('inf')  
        for i in xrange(10):
            clusters = kmeans(pick.point_list, 2)
            sum_sse = clusters[0].SSE() + clusters[1].SSE()
            if sum_sse < min_sum_sse:
                min_sum_sse = sum_sse
                best_clusters = []
                best_clusters.extend(clusters)
        list_clusters[picked_cluster_i] = best_clusters[0]
        list_clusters.append(best_clusters[1])

        cluster_colors = cm.rainbow(np.linspace(0,1,len(list_clusters)))        
        for i in xrange(len(list_clusters)):
            list_clusters[i].color = cluster_colors[i]

        if (SIMULATION):
            for i in xrange(len(list_clusters)):
                nppoints = np.asarray(list_clusters[i].point_list, dtype = float)
                plt.scatter(nppoints[0: , 0], nppoints[0: , 1], c=list_clusters[i].color)            
                plt.plot(list_clusters[i].centroid[0],list_clusters[i].centroid[1], '+', mew=10, ms=25,c='black')
                plt.plot(list_clusters[i].centroid[0],list_clusters[i].centroid[1], '+', mew=5, ms=20,c=list_clusters[i].color)
            plt.title("Bisecting K-Means")
            plt.show()

    return list_clusters            


if __name__ == '__main__':
    dataset = takeInput()
    list_clusters = bisectingKmeans(dataset, 12, SIMULATION = True)  
        
    for i in xrange(len(list_clusters)):
        nppoints = np.asarray(list_clusters[i].point_list, dtype = float)
        if(len(nppoints)!=0):
            plt.scatter(nppoints[0: , 0], nppoints[0: , 1], c=list_clusters[i].color)            
        plt.plot(list_clusters[i].centroid[0],list_clusters[i].centroid[1], '+', mew=10, ms=25,c='black')
        plt.plot(list_clusters[i].centroid[0],list_clusters[i].centroid[1], '+', mew=5, ms=20,c=list_clusters[i].color)
    plt.title('After bisecting kmeans')
    plt.show()  
    
    centroids = []
    for cl in list_clusters:
        centroids.append(cl.centroid)

    kmeans(dataset, 12, centroids, True)
    