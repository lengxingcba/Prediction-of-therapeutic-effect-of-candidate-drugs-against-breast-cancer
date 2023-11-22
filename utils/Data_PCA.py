import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

scaler = StandardScaler()   #标准化数据
data_futureSet=pd.read_csv("new_dataset.csv")
data_futureSet.drop(labels=["SMILES","pIC50"],axis=1,inplace=True)
scaler.fit(data_futureSet)
scaled=scaler.transform(data_futureSet)

pca=PCA(svd_solver="randomized",n_components=50)
pca.fit(scaled)
print(pca.explained_variance_ratio_.sum())


var = pca.explained_variance_ #percentage of variance explained
labels =[]
for i in range(1,51):
    labels.append("PC"+str(i))

plt.figure(figsize=(50,50))
plt.bar(labels,var,)
plt.xticks(fontsize=5)
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()