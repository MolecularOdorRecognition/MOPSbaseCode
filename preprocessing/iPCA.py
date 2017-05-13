"""
To observe the difference between PCA and iPCA datasets, this code is created to create the iPCA dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import pandas

# import data
X = pandas.read_csv("input/X_scaled.csv", header=None, low_memory=False, skiprows=1)
Y = pandas.read_csv("input/Y.csv", header=None, low_memory=False, skiprows=1)

# get iPCA
num = 337
ipca = IncrementalPCA(n_components=num)
X_ipca = ipca.fit_transform(X)

# convert to data frame and output to csv file
X_ipca = pandas.DataFrame(X_ipca)
X_ipca.to_csv("X_ipca.csv")
