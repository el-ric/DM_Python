from random import random
from sklearn.decomposition.pca import PCA

numberOfPoints = 50
initialNumberOfDimensions = 7
finalNumberOfDimensions = 2

points = []
for i in range(numberOfPoints):
    newPoint = []
    # generate the coordinates (between 0 and 100) for each dimension
    for k in range(initialNumberOfDimensions):
        newPoint.append(random() * 100)
    points.append(newPoint)

transformedPoints = PCA(finalNumberOfDimensions).fit_transform(points).tolist()

print("Initial number of dimensions:", len(points[0]))
print("Final number of dimensions:", len(transformedPoints[0]))


# optional: generate cluster assignments and centroids to directly
# use the plotClusters method from scatter_plot_example.py
from random import randint
from scatter_plot_example import plotClusters

clusters = 3

for point in transformedPoints:
    # generate the cluster assignment
    point.append(randint(1, clusters))

# generate random centroids with coordinates between 0 and 100
centroids = [[random() * 100, random() * 100] for i in range(clusters)]

plotClusters(centroids, transformedPoints)
