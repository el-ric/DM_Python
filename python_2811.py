import pandas
import random
import math

#filename = "GadgetManiacs_Cluster.xlsx"
filename = "DatasetClean_NotNormalized.csv"

dataDf = pandas.read_csv(filename)
data = dataDf.values.tolist()
lines = len(data)
columns = len(data[0])
print("\nNumber of lines:", lines)
print("Number of columns:", columns)

clusterColumns = []
clusterColumns=[2, 3, 4]
k = 3

kPoints = [[] for i in range(k)]
for i in range(k):
#need to change this to allow for non-consecutive columns
    kPoints[i] = data[random.randint(0, lines)][clusterColumns[0] : clusterColumns[len(clusterColumns)-1]+1]
#    print(kPoints[i])

distance = []
for i in range(lines):
    temp = data[i][clusterColumns[0] : clusterColumns[len(clusterColumns)-1]+1]
    for j in range(len(kPoints)):
        #calculate the euclidean distance from each point in the dataset to each cluster point
  
        distanceAux = []
        for l in range(len(temp)):
            distanceAux += [math.pow(temp[l] - kPoints[j][l], 2)]
      
        distanceAux[l] = math.sqrt(distanceAux[l])
        distance.append(distanceAux)
        
print(distance[10])