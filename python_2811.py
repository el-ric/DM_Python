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
        if(len(cluster)>0):
            transformedPoints = PCA(2).fit_transform(cluster).tolist()
            for point in transformedPoints:
                ax.scatter(point[0], point[1], color=colors[clusters.index(cluster)], s=5, marker=",")
                #ax.annotate(str(point[1]), (point[1] + 1, point[2] + 1))
                index = (index + 1) % len(colors)
    index = 0
    transformedPoints = PCA(2).fit_transform(centroids).tolist()
    print ("PCA centroids", transformedPoints)
    for point in transformedPoints:
        ax.scatter(point[0], point[1], color=colorsCluster[index], s=900, marker="x")
        ax.annotate("C" + str(index + 1), (point[0] + 2, point[1] + 2))
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
        initialCentroids[i] = selectedData[r][1:len(selectedData[r])-1]
        initialOffset = rightRange + 1
        rightRange = round(((lines/k) * (i + 2)),0)
    #
    print ("Initial centroid selected")
    print(initialCentroids)
    return initialCentroids
    
       
def selectInitialCentroid(selectedData):
    ##SELECTS RANDOM K CENTROIDS FROM THE NEW SET
    initialCentroids = [[] for i in range(k)]
    for i in range(k):
        r = random.randint(0, lines)
        #From the original dataset take away the first and last element (id of client and cluster)
        initialCentroids[i] = selectedData[r][1:len(selectedData[r])-1]

    
    print ("Initial centroids selected")
    print(initialCentroids)
    return initialCentroids

    
def selectInitialCentroidWeighted(selectedData):
    initialCentroids = [[0 for i in range (len(clusterColumns))] for j in range(k)]
    #From the original dataset take away the first and last element (id of client and cluster)                    
    for i in range(k):
        if i == 0:
        #select the first centroid randomly
            r = random.randint(0, lines)
            initialCentroids[i] = selectedData[r][1:len(selectedData[r])-1]
        else:
        ##SELECTS K-1 CENTROIDS FROM THE NEW SET WITH A WEIGHTED PROBABILITY
        #COMPUTE THE WEIGHT DISTANCE TO THE CENTROID
            initialDistance = [[] for i in range(lines)]
            smallestDistance = [0 for i in range(lines)]
            
            for client in range(lines):
                #calculate the distance to the previous selected centroids 
                #and select the shortest
                for cluster in range(k):
                    if initialCentroids[cluster] != [0,0,0]:
                       initialDistance[client].append(euclideanDistance(selectedData[client], initialCentroids[cluster]))
                       smallestDistance[client] = math.pow(min(initialDistance[client]),2)
            boundary = 0.0
            for client in range (lines):
                boundary += smallestDistance[client]
            r = random.uniform(0, boundary)
            cumulativeWeight = 0.0
            tempClient = -1
            for client in range(lines):
                if cumulativeWeight < r:  
                    cumulativeWeight += smallestDistance[client]
                else:                
                    tempClient = client
                    break
            initialCentroids[i] = selectedData[tempClient][1:len(selectedData[tempClient])-1]
    
    print ("Initial centroids selected")
    print(initialCentroids)
    return initialCentroids
    
    
def assignCluster(initialCentroids):
    ########DO THIS UNTIL CONVERGENCE!#########
    ##CALCULATE THE EUCLIDEAN DISTANCE FROM EACH POINT IN THE DATASET TO EACH CLUSTER POINT
    distance = [[] for i in range(lines)]
    #Every row / Client
    for client in range(lines):       
        #Every cluster needed
        for cluster in range(k):
            distance[client].append(euclideanDistance(selectedData[client], initialCentroids[cluster]))           
            #distance[client].append(manhattanDistance(selectedData[client], initialCentroids[cluster]))           
            #distance[client].append(weightedEuclideanDistance(selectedData[client], initialCentroids[cluster]))           
            #distance[client].append(MinkowskiDistance(selectedData[client], initialCentroids[cluster]))           
                      
            
    ##GETS THE CORRESPONDING CLUSTER FOR EACH CLIENT
    global reassignedClients
    reassignedClients = 0
    oldCluster = 0
    for client in range(lines):
        clusterNumber = distance[client].index(min(distance[client]))
        oldCluster =  selectedData[client][len(selectedData[client])-1]
        selectedData[client][len(selectedData[client])-1] = clusterNumber
        if(oldCluster != clusterNumber):
            reassignedClients += 1
            #print("Customer:",selectedData[clientID][0],"Old:",oldCluster," New:",clusterNumber,"Distance:",distance[clientID])
    print("reassignedClients")
    print(reassignedClients)

    #creates a list for each cluster with the client details --Easy to do math operations on separate lists.
    #THIS PART MIGHT BE DUPICATED; WE ARE GETTING THE CLUSTER ID ON TWO VARIABLES
    clusters = [[] for i in range(k)]  
    for client in selectedData:
        #Appends the client to the cluster list that corresponds WITHOUT the initial ID and the cluster number
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
                
                #print(client)
                #print(columnID)
               # print(client[columnID])
                sumVal[clusters.index(cluster)][columnID] += client[columnID]
                countVal[clusters.index(cluster)][columnID] += 1 
    print("Initial clustering", updatedCentroids)
    #print("sumVal",sumVal)
    #print("countVal",countVal)

    for cluster in clusters:
        for columnID in range(len(clusterColumns)):
            if(countVal[clusters.index(cluster)][columnID]>0):
                updatedCentroids[clusters.index(cluster)][columnID] = sumVal[clusters.index(cluster)][columnID] / countVal[clusters.index(cluster)][columnID]

    print("After",updatedCentroids)
    return updatedCentroids

    
def showResults():

    #plotClusters(centroids, clusters, runNumber)
    for i in range(k):
        print("Cluster {} has {} elements".format(i+1,len(clusters[i])))  
        

def showStatistics(centroids):
    for cluster in range(k):
        for column in range(len(clusterColumns)):
            print("Cluster {} in variable {} has an average of {}.".format(cluster+1,column+1,('%.2f'%(centroids[cluster][column]))))
        
def euclideanDistance(client, centroid):
    distanceAux = 0.0
    for column in range(len(clusterColumns)):
    # +1 FIX for the last column with the ID of the client
        distanceAux += math.pow(client[column + 1] - centroid[column], 2)
    distanceAux = math.sqrt(distanceAux)
    return distanceAux


def manhattanDistance(client, centroid):
    distanceAux = 0.0
    for column in range(len(clusterColumns)):
    # +1 FIX for the last column with the ID of the client
        distanceAux += abs(client[column + 1] - centroid[column])
    
    return distanceAux

def weightedEuclideanDistance(client, centroid):
    distanceAux = 0.0
    #Column weight must add to 1
    columnsWeight = [.33, .33, .33]
    for column in range(len(clusterColumns)):
    # +1 FIX for the last column with the ID of the client
        distanceAux += math.pow(client[column + 1] - centroid[column], 2) * columnsWeight[column]
    distanceAux = math.sqrt(distanceAux)
    return distanceAux

def MinkowskiDistance(client, centroid):
    distanceAux = 0.0
    r = 20
    #Column weight must add to 1
    for column in range(len(clusterColumns)):
    # +1 FIX for the last column with the ID of the client
        distanceAux += math.pow(math.pow(abs(client[column + 1] - centroid[column]), r),1/r)
    distanceAux = math.sqrt(distanceAux)
    return distanceAux

####GENERAL PARAMETERS ######
filename = "DatasetClean.csv"
clusterColumns = []
#clusterColumns=[3,4,1]
#clusterColumns=[65, 82, 102, 104]
clusterColumns = [65, 82, 102]
k = 3
numberIterations = 10
useUniformCentoirds = 0
runNumber = 1
stopChangePerIteration = 0.001


#Inicialization
print ("Running K-Means with K =  {} and {} variables ".format(k,len(clusterColumns)))
selectedData = readDataSet(filename, clusterColumns)

#centroids = selectInitialCentroid(selectedData)
centroids = selectInitialCentroidWeighted(selectedData)
reassignedClients = lines


#Move elements from centroids
while (reassignedClients/lines > stopChangePerIteration):
#    for i in range(1, numberIterations):
    print("Run ", runNumber)
    #Creates List with clusters and the clients in each cluster. 
    #Updates the original selectedData with the current number of cluster for the client.
    clusters = assignCluster(centroids)
    if(runNumber==1):
         showResults()
    centroids = updateCentroid(centroids,clusters)
    runNumber += 1
    
showStatistics(centroids)
showResults()

    

