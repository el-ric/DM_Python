import pandas
import random
import math
import matplotlib.pyplot as plot
from sklearn.decomposition.pca import PCA


def plotClustersNew(centroids, clusters,runNumber):
    colors = ["b", "g", "r","y","c","k","m","violet","aqua","forestgreen"]
    colorsCluster = ["g", "r","b","c","y","m","k","aqua","violet","b"]
#    markers = ["o", "o", "o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()
    i = 0
    ax.set_title('K-Means # {}'.format(runNumber))
    newData = []
    clusterNumber = []
    #clusterNumber = [[] for i in range(lines)]  
    for client in selectedData:
        newData.append(client[1:-1])
        clusterNumber.append(client[-1])
    
    for cluster in centroids:
        newData.append(cluster)

    transformedPoints = PCA(2).fit_transform(newData).tolist()
    target = len(transformedPoints) - len(centroids)
    for point in transformedPoints:
        if(i < target):
            ax.scatter(point[0], point[1], color=colors[clusterNumber[i]], s=5, marker=",")
        else:
            ax.scatter(point[0], point[1], color=colorsCluster[(i+1)  % len(colors)], s=250, marker="x")
        i += 1

    fig.canvas.draw()
    fig.show()


def plotClusters(centroids, clusters,runNumber):
    colors = ["b", "g", "r","y","c","k","m","violet","aqua","forestgreen"]
    colorsCluster = ["g", "r","b","c","y","m","k","aqua","violet","b"]
#    markers = ["o", "o", "o"]
#    markers = ["^", "s", ""]

    fig, ax = plot.subplots()
    index = 0
    ax.set_title('K-Means # {}'.format(runNumber))

    for cluster in clusters:
            if(len(cluster)>0):
                i = 1
                #Add the centroid to the set to create the graph on the same scale
                cluster.append(centroids[clusters.index(cluster)])
                transformedPoints = PCA(2).fit_transform(cluster).tolist()
                for point in transformedPoints:
                    if(i<len(transformedPoints)):
                        ax.scatter(point[0], point[1], color=colors[clusters.index(cluster)], s=5, marker=",")
                        #ax.annotate(str(point[1]), (point[1] + 1, point[2] + 1))
                        index = (index + 1) % len(colors)
                    else:
                        ax.scatter(point[0], point[1], color=colorsCluster[index], s=500, marker="D")
                        #delete centroid from the set
                        transformedPoints.pop(i-1)
                    i += 1
    

#    index = 0
#    transformedPoints = PCA(2).fit_transform(centroids).tolist()
#    print ("PCA centroids", transformedPoints)
#    for point in transformedPoints:
#        ax.scatter(point[0], point[1], color=colorsCluster[index], s=900, marker="x")
#        ax.annotate("C" + str(index + 1), (point[0] + 2, point[1] + 2))
#        index = (index + 1) % len(colors)

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

    
def selectInitialCentroidWeighted(selectedData, k, finalFunction):
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
                       if(selectedDistance == "Euclidean"):
                          initialDistance[client].append(euclideanDistance(selectedData[client], initialCentroids[cluster]))
                       elif(selectedDistance == "Manhattan"):
                          initialDistance[client].append(manhattanDistance(selectedData[client], initialCentroids[cluster]))
                       elif(selectedDistance == "Minkowski"):
                          initialDistance[client].append(MinkowskiDistance(selectedData[client], initialCentroids[cluster]))
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
    
    
    if finalFunction: 
        print ("Initial centroids selected")
        print(initialCentroids)
    return initialCentroids
    
    
def assignCluster(initialCentroids, k):
    ########DO THIS UNTIL CONVERGENCE!#########
    ##CALCULATE THE EUCLIDEAN DISTANCE FROM EACH POINT IN THE DATASET TO EACH CLUSTER POINT
    distance = [[] for i in range(lines)]
    #Every row / Client
    for client in range(lines):       
        #Every cluster needed
        for cluster in range(k):
            if(selectedDistance == "Euclidean"):
                distance[client].append(euclideanDistance(selectedData[client], initialCentroids[cluster]))           
            elif(selectedDistance == "Manhattan"):
                distance[client].append(manhattanDistance(selectedData[client], initialCentroids[cluster]))           
            elif(selectedDistance == "Minkowski"):       
                distance[client].append(MinkowskiDistance(selectedData[client], initialCentroids[cluster]))           
                      
    ##GETS THE CORRESPONDING CLUSTER FOR EACH CLIENT
    
    reassignedClients = 0
    oldCluster = 0
    for client in range(lines):
        clusterNumber = distance[client].index(min(distance[client]))
        oldCluster =  selectedData[client][len(selectedData[client])-1]
        selectedData[client][len(selectedData[client])-1] = clusterNumber
        if(oldCluster != clusterNumber):
            reassignedClients += 1
            #print("Customer:",selectedData[clientID][0],"Old:",oldCluster," New:",clusterNumber,"Distance:",distance[clientID])
    #print("ReassignedClients: {}".format(reassignedClients))
    

    #creates a list for each cluster with the client details --Easy to do math operations on separate lists.
    clusters = [[] for i in range(k)]  
    for client in selectedData:
        #Appends the client to the cluster list that corresponds WITHOUT the initial ID and the cluster number
        clusters[client[len(client) - 1]].append(client[1:len(client) - 1])
    return clusters, reassignedClients

    
    
def updateCentroid(updatedCentroids,clusters, k):
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
    #print("Initial clustering", updatedCentroids)
    #print("sumVal",sumVal)
    #print("countVal",countVal)

    for cluster in clusters:
        for columnID in range(len(clusterColumns)):
            if(countVal[clusters.index(cluster)][columnID]>0):
                updatedCentroids[clusters.index(cluster)][columnID] = sumVal[clusters.index(cluster)][columnID] / countVal[clusters.index(cluster)][columnID]

    #print("After",updatedCentroids)
    return updatedCentroids


def intraclusterVariability(centroids, clusters, finalFunction):
    distanceAux = 0.0
    intraclusterVariability = []
    for i in range(len(centroids)):
        for client in range(len(clusters[i])):
            tempClient = clusters[i][client]
            for column in range(len(clusterColumns)):
                distanceAux += math.pow(tempClient[column] - centroids[i][column], 2)
        intraclusterVariability.append(distanceAux)
    sumICV = 0.0
    for i in range(len(centroids)):
        sumICV += intraclusterVariability[i]
        if finalFunction:
            print("Cluster {} has a intracluster variability of {}.".format(i+1, ('%.2f'%intraclusterVariability[i])))
    return sumICV
    
        
def clusterAverage(centroids):
    for cluster in range(k):
        for column in range(len(clusterColumns)):
            print("Cluster {} in variable {} has an average of {}.".format(cluster+1,column+1,('%.2f'%(centroids[cluster][column]))))    
    
def clusterDistribution(clusters):
    for i in range(k):
        print("Cluster {} has {} elements".format(i+1,len(clusters[i])))  
        
        
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
    
def MinkowskiDistance(client, centroid): 
    distanceAux = 0.0 
    r = 20 
    #Column weight must add to 1 
    for column in range(len(clusterColumns)): 
    # +1 FIX for the last column with the ID of the client 
        distanceAux += math.pow(math.pow(abs(client[column + 1] - centroid[column]), r),1/r) 
    return distanceAux 

#def weightedEuclideanDistance(client, centroid):
#    distanceAux = 0.0
#    #Column weight must add to 1
#    columnsWeight = [.33, .33, .33]
#    for column in range(len(clusterColumns)):
    # +1 FIX for the last c                                          

    
def findAK():
    terminationRule = 12
    oldSumICV = 0
    newSumICV = 0
    stopChangePerIteration = 0
    plateauFlag = False
    for k in range (2, terminationRule):
        oldSumICV = newSumICV
        centroids = selectInitialCentroidWeighted(selectedData, k, False)
        reassignedClients = 1
        while (reassignedClients/lines > stopChangePerIteration):
            clusters, reassignedClients = assignCluster(centroids, k)
            centroids = updateCentroid(centroids,clusters, k)
        newSumICV = intraclusterVariability(centroids, clusters, False)
        oldAvgICV = oldSumICV/(k-1)
        newAvgICV = newSumICV/k
        print("Testing with K={}".format(k))
        if (oldAvgICV != 0) and ((newSumICV > oldSumICV) and (newAvgICV/oldAvgICV > .80)): 
            print("\nSelected number of k: {} ".format(k-1))
            print("K={} intracluster variability sum: {}".format(k-1,'%.2f'%oldSumICV))
            print("K={} intracluster variability sum: {}".format(k, '%.2f'%newSumICV))
            print("K={} intracluster variability average: {}".format(k-1,'%.2f'%oldAvgICV))
            print("K={} intracluster variability average: {}".format(k, '%.2f'%newAvgICV))
            print("K={} intracluster variability average/K={} intracluster variability average: {}".format(k, k-1, '%.2f'%(newAvgICV / oldAvgICV)))
            return k-1
        elif (oldAvgICV != 0) and (newAvgICV/oldAvgICV > .80 and plateauFlag):
            print("Reached a plateau.")
            print("\nSelected number of k: {} ".format(k-2))
            return k-2
        if (oldAvgICV != 0) and (newAvgICV/oldAvgICV > .80):
            plateauFlag = True
    return -1   
    
def automaticallyFindK():
    numberOfAttempts = 6
    suggestedK = []
    acceptableNClusters = 10
    countSuggestedK = []
    finalK = 0
    for i in range (numberOfAttempts):
        print("\nAttempt #{} to find k.".format(i+1))
        suggestedK.append(findAK())
    for i in range (2, acceptableNClusters+2):
        countSuggestedK.append(suggestedK.count(i))
    for i in range (len(countSuggestedK)):
        if(countSuggestedK[i] > 0):
            print("K = {} was suggested {} times.".format(i+2, countSuggestedK[i]))
    countSuggestedK.append(suggestedK.count(-1))
    if(countSuggestedK[acceptableNClusters] > 0):
        print("Unable to find an acceptable k {} times.".format(i+2, countSuggestedK[i]))
    print("Suggested K")
    print(suggestedK)
    finalK = countSuggestedK.index(max(countSuggestedK)) + 2
    
    print("Final K")
    print(finalK)
    return finalK
    
def kMeans(k):
    stopChangePerIteration = 0
    centroids = selectInitialCentroidWeighted(selectedData, k, True)
    reassignedClients = 1
    iterationNumber = 1
    while (reassignedClients/lines > stopChangePerIteration):
            clusters, reassignedClients = assignCluster(centroids, k)
            centroids = updateCentroid(centroids,clusters, k)
            if(iterationNumber==1):
                print("\nResults of the first iteration:")
                clusterDistribution(clusters)
            iterationNumber += 1
    print("\nTermination condition reached.")
    print("\nResults of the last iteration (iteration {})".format(iterationNumber))
    sumICV = intraclusterVariability(centroids, clusters, True)
    print("Sum of intracluster variability: {}".format('%.2f'%sumICV))
    print("Average intracluster variability: {}.".format('%.2f'%(sumICV/k)))
    clusterDistribution(clusters)   
    clusterAverage(centroids) 
    plotClustersNew(centroids, clusters, iterationNumber)

####GENERAL PARAMETERS ######
filename = "DatasetClean.csv"
clusterColumns = []
#clusterColumns=[3,4,1]
#clusterColumns=[65, 82, 102, 104]
clusterColumns = [82, 102, 104]
numberIterations = 10
stopChangePerIteration = 0

selectedDistance = "Euclidean"
#selectedDistance = "Manhattan"
#selectedDistance = "Minkowski"

def tests():
    terminationRule = 11
    newSumICV = 0
    stopChangePerIteration = 0
    for k in range (2, terminationRule):
        centroids = selectInitialCentroidWeighted(selectedData, k, False)
        reassignedClients = 1
        while (reassignedClients/lines > stopChangePerIteration):
            clusters, reassignedClients = assignCluster(centroids, k)
            centroids = updateCentroid(centroids,clusters, k)
        newSumICV = intraclusterVariability(centroids, clusters, False)
        print(newSumICV)
        

#Read data
selectedData = readDataSet(filename, clusterColumns)
tests()

#Select K                
#k = 4
#k = automaticallyFindK()
#if k != -1:
#    print ("\nRunning K-Means with K = {} and {} variables ".format(k,len(clusterColumns)))
#    kMeans(k)
#else:
#    print("Could not determine K. Please try again or manually define k.")

    