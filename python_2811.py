import pandas
import random
import math

#filename = "GadgetManiacs_Cluster.xlsx"
#filename = "CleanSupermarket.xlsx"
#
#dataDf = pandas.read_excel(filename)
#data = dataDf.values.tolist()
lines = len(data)
columns = len(data[0])
#print("\nNumber of lines:", lines)
#print("Number of columns:", columns)

clusterColumns = []
clusterColumns=[2, 4, 6, 3]
k = 3


#SELECTS THE SAMPLE WITH ONLY THE COLUMNS TO CLUSTER
selectedData = [[] for i in range(lines)]
for i in range(lines):
    selectedData[i] = [data[i][0]]
    for j in range(len(clusterColumns)):
        selectedData[i].append(data[i][clusterColumns[j]])

        
#SELECTS FROM THE SELECTED DATA K CENTROIDS
initialCentroids = [[] for i in range(k)]
for i in range(k):
    r = random.randint(0, lines)
    initialCentroids[i] = selectedData[r]

#######DO THIS UNTIL CONVERGENCE!#########
#CALCULATE THE EUCLIDEAN DISTANCE FROM EACH POINT IN THE DATASET TO EACH CLUSTER POINT
distance = [[] for i in range(lines)]
for i in range(lines):       
    distanceAux = 0.0
    for j in range(k):
        for l in range(1, len(clusterColumns)):
            distanceAux += math.pow(selectedData[i][l] - initialCentroids[j][l], 2)
        distanceAux = math.sqrt(distanceAux)
        distance[i].append(distanceAux)

#CHECKS THE INDEX OF THE SMALLEST DISTANCE
clusterAttribution = []
for i in range(lines):
    minDistance = distance[i].index(min(distance[i]))
    clusterAttribution.append(minDistance)

    
#ITERATES OVER THE LIST OF CLUSTERS AND UPDATES CENTROIDS
updatedCentroids = [[0 for l in range (len(clusterColumns))] for i in range (k)]

for i in range(lines):
    for j in range(k): 
        if clusterAttribution[i] == j:
            for l in range (len(clusterColumns)):
                updatedCentroids[j][l] += selectedData[i][l+1]

print("Sum: ", updatedCentroids)

#calculates the number of individuals in each cluster
numberOfIndividuals = []
for i in range(k):
    temp = clusterAttribution.count(i)
    numberOfIndividuals.append(temp)

print(numberOfIndividuals)

#calculates the average value for each position of each centroid
for i in range(k):
    for j in range(len(clusterColumns)):
        updatedCentroids[i][j] = updatedCentroids[i][j]/numberOfIndividuals[i]

print("Average: ", updatedCentroids)

##we need to store the old centroid values so that we can compare the new centroids and
##the old centroids to check for convergence - something like, if this is the first time 
##the code runs, the old centroids is the initial random points; if this is not the first
##time, the old centroids are the previous centroids.







