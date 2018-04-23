from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import random
from functools import reduce, partial
from operator import add

class KMeans:
    """good old class based solution"""
    def __init__(self, k):
        self.k = k
        self.means = [None for _ in range(k)]

    def fit(self, points, num_iters=10):
        assignments = [None for _ in points]
        self.means = random.sample(list(points), self.k)
        for _ in range(num_iters):
            for i, point in enumerate(points):
                assignments[i] = self.predict(point)
            for j in range(self.k):
                cluster = [p for p, c in zip(points, assignments) if c == j]
                self.means[j] = list(map(lambda x: x / len(cluster), reduce(partial(map, add), cluster)))

    def predict(self, point):
        d_min = float('inf')
        for j, m in enumerate(self.means):
            d = sum((m_i - p_i)**2 for m_i, p_i in zip(m, point))
            if d < d_min:
                prediction = j
                d_min = d
        return prediction

def run_kmeans(points, k=3):
    model = KMeans(k)
    model.fit(points, num_iters=100)
    assignments = [model.predict(point) for point in points]

    for x, y in model.means:
        plt.plot(x, y, marker='*', markersize=20, color='Black')

    for j, color in zip(range(k),
                      ['r', 'g', 'b', 'm', 'c']):
        cluster = [p
                   for p, c in zip(points, assignments)
                   if j == c]
        xs, ys = zip(*cluster)
        plt.scatter(xs, ys, color=color)

    plt.show()

#Generate some data
random.seed(42)
points = np.random.random((100,2))

run_kmeans(points, 5)