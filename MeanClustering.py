import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import pandas as pnd

p = pnd.read_csv("./group2.csv")  # TESTED
d = p.to_numpy()


def normalize(a: np.array):  # TESTED
    l = a.transpose()
    i = 0
    s = a.shape
    while i < s[1]:
        l[i] = (l[i] - l[i].mean()) / l[i].std(ddof=1)
        i = i + 1
    l = l.transpose()
    return l


def K_mean_cluster(k: int, data: np.array):  # TESTED complixity O(k*n)
    data = normalize(data)
    s = data.shape
    points = []
    for i in range(k):
        points.append(data[np.random.randint(s[0])])
    points = np.array(points)
    delta = 0.5
    return Cluster(k, data, points, delta, s)


def Cluster(k: int, data: np.array, points, delta: np.double, s):  # TESTED
    current = [[] for _ in range(k)]
    for i in range(s[0]):
        min = np.inf
        index = -1
        for j in range(k):
            dist = lg.norm(data[i] - points[j])
            if dist < min:
                min = dist
                index = j
        current[index].append(data[i].tolist())
    anothercall = False
    for j in range(k):
        temp = np.array(current[j])
        temp = temp.transpose()
        y = temp.shape
        if y == (0,):
            continue
        m = [None] * y[0]
        for i in range(y[0]):
            m[i] = temp[i].mean()
        mean = np.array(m)
        dist = lg.norm(mean - points[
            j])  # NOTE Euclidean distance is the l2 norm, and the default value of the ord parameter in numpy.linalg.norm is 2.
        if dist > delta:
            points[j] = mean
            anothercall = True
    if anothercall:
        return Cluster(k, data, points, delta, s)
    else:
        return points, current


def pca(data: np.array):  # TESTED
    data = normalize(data)
    CovMatrix = np.cov(data, rowvar=False)
    EigenValues, EigenVectors = lg.eig(CovMatrix)
    idx = np.argsort(EigenValues)
    EigenValues = EigenValues[idx]
    EigenVectors = EigenVectors[idx]
    data = np.matmul(data, EigenVectors)
    data = data.transpose()
    data = data[-2:]
    data = data.transpose()
    return data


k = eval(input("Enter the k value : "))
hello = pca(d)
centers, clusters = K_mean_cluster(k, hello)
for i in range(k):
    percentage = len(clusters[i]) / 100000
    print("Percentage of data in cluster ", i, "is ", percentage)
    hello1 = np.array(clusters[i])
    hello1 = hello1.transpose()
    x1 = hello1[0]
    y1 = hello1[1]
    plt.scatter(x1, y1)
    plt.scatter(centers[i, 0], centers[i, 1])
plt.show()
