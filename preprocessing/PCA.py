"""
To create the dataset preprocessed by the PCA method.
"""
import pandas
from sklearn.decomposition import PCA

#import data
X = pandas.read_csv("X.csv", header=None, low_memory=False, skiprows=1)

#convert data to Matrix form
X = X.as_matrix()

pca = PCA(n_components=300)
pca.fit(X)
print(pca.explained_variance_ratio_)

pca.n_components = 175
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

# convert to DataFrame
X_reduced = pandas.DataFrame(X_reduced)

# Save data in CSV
X_reduced.to_csv("X_reduced.csv")
