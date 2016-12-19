import pandas
import random
import math
import matplotlib.pyplot as plot
from sklearn.decomposition.pca import PCA

def plotClusters(centroids, clusters,runNumber):
    colors = ["b", "g", "r","y","c","k","m","violet","aqua","forestgreen"]
    colorsCluster = ["g", "r","b","y","c","k","m","violet","aqua","forestgreen"]
#    markers = ["o", "o", "o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()
    index = 0
    ax.set_title('K-Means # {}'.format(runNumber))
    for cluster in clusters:
        for client in cluster:
            ax.scatter(client[1], client[2], color=colors[clusters.index(cluster)], s=5, marker=",")
            #ax.annotate(str(point[1]), (point[1] + 1, point[2] + 1))
            index = (index + 1) % len(colors)
    index = 0
    for centroid in centroids:
        ax.scatter(centroid[1], centroid[2], color=colorsCluster[index], s=900, marker="x")
        ax.annotate("C" + str(index + 1), (centroid[1] + 2, centroid[2] + 2))
        index = (index + 1) % len(colors)

    fig.canvas.draw()
    fig.show()


def readDataSet(filename, clusterColumns):
    print("Reading file...")    
    dataDf = pandas.read_csv(filename)
    data = dataDf.values.tolist()
    global lines 
    lines = len(data)
    columns = len(data[0])
    initialCluster = 0
    print("Number of lines:", lines)
    print("Number of columns:", columns)   
    #SELECTS THE SAMPLE WITH ONLY THE COLUMNS TO CLUSTER
    selectedData = [[] for i in range(lines)]    
    for i in range(lines):
        selectedData[i] = [data[i][0]]
        for j in range(len(clusterColumns)):
            selectedData[i].append(data[i][clusterColumns[j]])
        selectedData[i].append(initialCluster)
    
    return selectedData

    
def selectInitialCentoidUniform(selectedData):
    ##SELECTS RANDOM K CENTROIDS FROM THE NEW SET
    initialCentroids = [[] for i in range(k)]
    initialOffset = 0
    rightRange = round(lines/k,0)
    print("Running with uniform initial clusters")
    print("Intervals for initial clusters")
    for i in range(k):
        r = random.randint(initialOffset,rightRange )
        print("Client number from {} to {}".format(initialOffset,rightRange))
        initialCentroids[i] = selectedData[r]
        if (offsetIdClient == 0):
            initialCentroids[i].pop(0)
        initialOffset = rightRange + 1
        rightRange = round(((lines/k) * (i + 2)),0)
    #
    print ("Initial centroid selected")
    print(initialCentroids)
    return initialCentroids
    
    
    
def selectInitialCentoid(selectedData):
    ##SELECTS RANDOM K CENTROIDS FROM THE NEW SET
    initialCentroids = [[] for i in range(k)]
    for i in range(k):
        r = random.randint(0, lines)
        initialCentroids[i] = selectedData[r]
        if (offsetIdClient == 0):
            initialCentroids[i].pop(0)
    #
    print ("Initial centroid selected")
    print(initialCentroids)
    return initialCentroids
    
def assignCluster(initialCentroids):
    ########DO THIS UNTIL CONVERGENCE!#########
    ##CALCULATE THE EUCLIDEAN DISTANCE FROM EACH POINT IN THE DATASET TO EACH CLUSTER POINT
    distance = [[] for i in range(lines)]
    #Every row / Client
    for client in range(lines):       
        distanceAux = 0.0
        #Every cluster needed
        for cluster in range(k):
            #All the columns in the centroids
            for column in range(1, len(clusterColumns)):
                distanceAux += math.pow(selectedData[client][column] - initialCentroids[cluster][column], 2)
            distanceAux = math.sqrt(distanceAux)
            distance[client].append(distanceAux)
    #Distance from one point to all the others
            
    ##GETS THE CORRESPONDING CLUSTER FOR EACH CLIENT
    global reassignedClients
    reassignedClients = 0
    oldCluster = 0
    for clientID in range(lines):
        clusterNumber = distance[clientID].index(min(distance[clientID]))
        oldCluster =  selectedData[clientID][len(selectedData[clientID])-1]
        selectedData[clientID][len(selectedData[clientID])-1] = clusterNumber
        if(oldCluster != clusterNumber):
            reassignedClients += 1
    print("reassignedClients")
    print(reassignedClients)
    #creates a list for each cluster with the client details --Easy to do math operations on separate lists.
    #THIS PART MIGHT BE DUPICATED; WE ARE GETTING THE CLUSTER ID ON TWO VARIABLES
    clusters = [[] for i in range(k)]  
    for client in selectedData:
        clusters[client[len(client) - 1]].append(client[1:len(client) - 1])

    return clusters

    
    
def updateCentroid(updatedCentroids,clusters):
    sumVal = [[0 for l in range (len(clusterColumns))] for i in range (k)]
    countVal =  [[0 for l in range (len(clusterColumns))] for i in range (k)]
    for cluster in clusters:
        for client in cluster:
            # -2 because we have 2 extra columns
            for columnID in range(len(clusterColumns)):
                # +1 offsetIdClient because the first element of the client is the ID.
                
                print(client)
                print(columnID)
                print(client[columnID])
                sumVal[clusters.index(cluster)][columnID] += client[columnID]
                countVal[clusters.index(cluster)][columnID] += 1 
    print("Initial clustering", updatedCentroids)
    print("sumVal",sumVal)
    print("countVal",countVal)

    for cluster in clusters:
        for availableColumns in range(len(clusterColumns)):
            if(countVal[clusters.index(cluster)][availableColumns]>0):
                #+1 offsetIdClient because the first element of the client is the ID.
                updatedCentroids[clusters.index(cluster)][availableColumns + offsetIdClient] = sumVal[clusters.index(cluster)][availableColumns] / countVal[clusters.index(cluster)][availableColumns]

    print("After",updatedCentroids)
    return updatedCentroids

    
def showResults():

    #plotClusters(centroids, clusters,runNumber)
    for i in range(k):
        print("Cluster {} has {} elements".format(i+1,len(clusters[i])))  
        
            
####GENERAL PARAMETERS ######
filename = "DatasetClean.csv"
clusterColumns = []
#clusterColumns=[3,4,1]
#We need to set this to 1 if we are reading the ID of the client.
offsetIdClient = 0
clusterColumns=[46, 81,83]
k = 3
numberIterations = 10
useUniformCentoirds = 1
runNumber = 1
stopChangePerIteration = 0.001


#Inicialization
print ("Running K-Means with K =  {} and {} variables ".format(k,len(clusterColumns)-1))
selectedData = readDataSet(filename, clusterColumns)


if(useUniformCentoirds == 1):
    centroids = selectInitialCentoidUniform(selectedData)
else:
    centroids = selectInitialCentoid(selectedData)
reassignedClients = lines


#Move elements from centroids
while (reassignedClients/lines > stopChangePerIteration):
#for i in range(1, numberIterations):
    print("Run ", runNumber)
    #Creates List with clusters and the clients in each cluster. 
    #Updates the original selectedData with the current number of cluster for the client.
    clusters = assignCluster(centroids)
    if(runNumber==1):
         showResults()
    centroids = updateCentroid(centroids,clusters)
    runNumber += 1
    
showResults()
    

