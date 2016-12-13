import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib

def takeInput(filename="bisecting.txt", delimiter="\t"):
    point_list = []
    with open(filename, "r") as f:
        line = f.readline()
        print line
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
        self.point_list = point_list
        if (len(point_list)==0):
            self.centroid = centroid_tuple
        else:        
            self.centroid = self.updateCentroid()

    def updateCentroid(self):
        if (len(self.point_list)!=0):
            sumx = 0
            sumy = 0
            for tup in self.point_list:
                sumx += tup[0]
                sumy += tup[1]
            sumx /= len(self.point_list)
            sumy /= len(self.point_list)
            self.centroid = (sumx, sumy)
        return self.centroid

    def SSE(self):
        sse = 0
        for tup in point_list:
            x = tup[0]
            y = tup[1]
            sse += (self.centroid[0] - tup[0])**2 + (self.centroid[1] - tup[1])**2 
        return sse
    
    def distance(self, point):
        return ((self.centroid[0] - point[0])**2 + (self.centroid[1] - point[1])**2)

import numpy as np

def kmeans(point_list, k):
    nplist = np.asarray(point_list , dtype='float')

    xlist = nplist[0: , 0]  # [1, 2, 3]
    ylist = nplist[0: , 1]  # [2, 4, 6]
    label = np.zeros(len(xlist), dtype='int32') # [0, 0, 0]

    #selecting k points as initial centroid
    maxx, maxy = np.max(nplist, axis = 0)
    minx, miny = np.min(nplist, axis = 0)
    centroids = []    
    for i in range(k):
        ranx = np.random.uniform(minx, maxx)
        rany = np.random.uniform(miny, maxy)
        centroids.append((ranx, rany))
    centroids =  np.asarray(centroids , dtype='float')
    #plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, color='r' )

    t = np.arange(len(xlist))
    plt.scatter(xlist, ylist, c=t)
    cluster_colors =(cm.rainbow(np.linspace(0,1,k)))
    #plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, c=cluster_colors )
    plt.show()

    # initial k clusters
    clusters = []

    for i in range(k):
        c = Cluster([], centroids[i])
        c.color = cluster_colors[i]
        clusters.append(c) 

    # repeat
    # form k clusters by assigning all points to the closest centroid
    # recompute the centroid
    # until the centroids dnt change
    breakflag = False
    for iter in range(20):   
        if breakflag:
            break         
        for ip in range(len(point_list)):
            point = point_list[ip]
            dists = []        
            for i in range(k):
                dists.append(clusters[i].distance(point))
            min_dist_k = np.argmin(dists)
            label[ip] = min_dist_k
            clusters[min_dist_k].point_list.append(point)

        new_centroids = []
        for i in range(k):        
            new_centroids.append(clusters[i].updateCentroid())
        #centroids = np.asarray(centroids, dtype=float)
        new_centroids = np.asarray(new_centroids, dtype=float)
        if (np.allclose(centroids, new_centroids)):
            breakflag = True    
        centroids = new_centroids

        
        #for i in range(k):
        #    col = [1,2,2,2]
        #    col[0] = Blues(clusters[i].color)[0] * 255
        #    col[1] = Blues(clusters[i].color)[1] * 255
        #    col[2] = Blues(clusters[i].color)[2] * 255
        #    col[3] = Blues(clusters[i].color)[3] * 255
        #    col = Blues(clusters[i].color)
        #    t2.append(col)
       # plt.plot(3, 5, mew = 5, ms = 20, color = t[0])
        #plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, color='r' )
        
        
    t=(cm.rainbow(np.linspace(0,1,len(xlist))))
    for i in range(len(xlist)):
        t[i] = clusters[label[i]].color
    #for i,tt in zip(range(k), t):
    plt.scatter(xlist,ylist, c=t)
    #plt.scatter(xlist, ylist, cmap=t)
    for i,cc in zip(range(k), cluster_colors):
        plt.plot(centroids[i][0],centroids[i][1], '+', mew=10, ms=25,c='black')
        plt.plot(centroids[i][0],centroids[i][1], '+', mew=5, ms=20,c=cc)
    plt.show()
    #print x, y
#kmeans(l, 3)

#from matplotlib import animation

## create a simple animation
#fig = plt.figure()
#ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
#centroids = initialize_centroids(points, 3)

#def init():
#    return

#def animate(i):
#    global centroids
#    closest = closest_centroid(points, centroids)
#    centroids = move_centroids(points, closest, centroids)
#    ax.cla()
#    ax.scatter(points[:, 0], points[:, 1], c=closest)
#    ax.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
#    return 

#animation.FuncAnimation(fig, animate, init_func=init,
#                        frames=10, interval=200, blit=True)

if __name__ == '__main__':
    #dataset = takeInput("input.txt", " ")
    dataset = takeInput()
    kmeans(dataset, 7)
    '''
    nplist = np.asarray(dataset , dtype='float')
    xlist = nplist[0: , 0]
    ylist = nplist[0: , 1]
    meanx = np.mean(xlist)
    meany = np.mean(ylist)
    t = np.arange(len(dataset))
    plt.plot([meanx], [meany], '+', mew=10, ms=20)
    plt.scatter(xlist, ylist, c=t)
    plt.show()
    '''
    
    #plt.plot(x, y)
    #plt.show()


