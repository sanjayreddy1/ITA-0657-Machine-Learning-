import math
from collections import Counter

# Sample dataset
# Format: [feature1, feature2, label]
dataset = [
    [2, 4, 'A'],
    [4, 6, 'A'],
    [4, 4, 'A'],
    [6, 2, 'B'],
    [6, 4, 'B'],
    [8, 4, 'B']
]

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# KNN Algorithm
def knn(dataset, test_point, k):
    distances = []

    # Calculate distance between test_point and all dataset points
    for data in dataset:
        features = data[:-1]  # Extract features
        label = data[-1]      # Extract label
        dist = euclidean_distance(test_point, features)
        distances.append((dist, label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    neighbors = distances[:k]

    # Get the most common class label
    labels = [label for (_, label) in neighbors]
    prediction = Counter(labels).most_common(1)[0][0]

    return prediction

# Test data point
test_point = [5, 5]
k = 3
test_point = [4, 5]
k = 3
test_point = [5, 3]
k = 3


result = knn(dataset, test_point, k)
print("Test Point:", test_point)
print("Predicted Class:", result)