import pandas as pd
import numpy as np
import sklearn.neural_network as MLPnetwork
import  sklearn.cross_decomposition as traintest_split
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
# iris = load_iris()
# print(iris.data[:3])
# print(iris.data[15:18])
# # we extract only the lengths and widthes of the petals:
# X = iris.data[:, (2, 3)]
# print(iris.data[37:40])
# print(iris.target)
#
# y = (iris.target == 0).astype(np.int8)
# print(y)
#
# p = Perceptron(random_state=42,max_iter=10)
# p.fit(X, y)
#
# values = [[1.5, 0.1], [1.8, 0.4], [1.3,0.2]]
#
# for value in X:
#     pred = p.predict([value])
#     print([pred])
#

print(np.arange(1,11,1))
