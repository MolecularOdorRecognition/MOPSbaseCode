import pandas
from sklearn import preprocessing

# import data
X = pandas.read_csv("X.csv", header=None, low_memory=False, skiprows=1)
Y = pandas.read_csv("Y.csv", header=None, low_memory=False, skiprows=1)

# scale data
X = X.as_matrix()
Y = Y.as_matrix()
X_scaled = preprocessing.scale(X)
Y_scaled = preprocessing.scale(Y)

# convert to data frame
X_scaled = pandas.DataFrame(X_scaled)
Y_scaled = pandas.DataFrame(Y_scaled)

# send to csv
X_scaled.to_csv("X_scaled.csv")
Y_scaled.to_csv("Y_scaled.csv")
