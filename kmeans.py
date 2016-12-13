import numpy as np
import matplotlib.pyplot as plt

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
        self.point_list = np.asarray(point_list, dtype='float')
        if (len(point_list)==0):
            self.centroid = centroid_tuple
        else:        
            self.centroid = self.updateCentroid()

    def updateCentroid(self):
        sumx = 0
        sumy = 0
        for tup in self.point_list:
            sumx += tup[0]
            sumy += tup[1]
        sumx /= len(point_list)
        sumy /= len(point_list)
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
        return ((self.centroid[0] - point[0])**2 + (self.centroid[1] - point[1])**2)**(0.5)

import numpy as np

def kmeans(point_list, k):
    nplist = np.asarray(point_list , dtype='float')
    label = np.zeros(len(nplist))
    maxx, maxy = np.max(nplist, axis = 0)
    minx, miny = np.min(nplist, axis = 0)

    #x = np.reshape(nplist, (len(nplist)/4))
    xlist = nplist[0: , 0]
    ylist = nplist[0: , 1]
    #print x


    #selecting k points as initial centroid
    centroids = []
    
    for i in range(k):
        ranx = np.random.uniform(minx, maxx)
        rany = np.random.uniform(miny, maxy)
        centroids.append((ranx, rany))
    centroids =  np.asarray(centroids , dtype='float')
    plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, color='r' )
    plt.scatter(xlist, ylist, c=t)
    plt.show()

    # repeat
    # form k clusters by assigning all points to the closest centroid
    # recompute the centroid
    # until the centroids dnt change
    clusters = []
    
    for i in range(k):
        c = Cluster([], centroids[i])
        clusters.append(c)
    for ip in range(len(point_list)):
        point = point_list[ip]
        dists = []        
        for i in range(k):
            dists.append(clusters[i].distance(point))
        min_dist_k = np.argmin(dists)
        label[ip] = min_dist_k
    for i in range(len(label)):
        lbl = label[i]
        clusters[lbl].point_list.append(point_list[i])
    for i in range(k):        
        centroids[i] = clusters[i].updateCentroid()
    plt.plot(centroids[0: , 0], centroids[0: , 1], '+', mew=5, ms=20, color='r' )
    plt.scatter(xlist, ylist, c=t)
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
    dataset = takeInput("input.txt", " ")
    kmeans(dataset, 3)
    nplist = np.asarray(dataset , dtype='float')
    xlist = nplist[0: , 0]
    ylist = nplist[0: , 1]
    meanx = np.mean(x)
    meany = np.mean(y)
    t = np.arange(len(dataset))
    plt.plot([meanx], [meany], '+', mew=10, ms=20)
    plt.scatter(xlist, ylist, c=t)
    plt.show()
    #plt.plot(x, y)
    #plt.show()


