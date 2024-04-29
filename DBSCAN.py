import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("winequality-white.csv", sep=';')
data = data.drop_duplicates()

#features for clustering
X = data.iloc[:, :-1] 

#scaling
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

#exp 1 (default parameters)
dbscan1 = DBSCAN()
labels1 = dbscan1.fit_predict(Xscaled)

#exp 2 (adjusted parameters)
dbscan2 = DBSCAN(eps=1, min_samples=5)
labels2 = dbscan2.fit_predict(Xscaled)

#exp 3 (adjusted parameters)
dbscan3 = DBSCAN(eps=2, min_samples=10)
labels3 = dbscan3.fit_predict(Xscaled)

#histograms
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(labels1, color='skyblue')
plt.title('Experiment 1')
plt.xlabel('Cluster')
plt.ylabel('Number of samples')

plt.subplot(1, 3, 2)
plt.hist(labels2, color='lightgreen')
plt.title('Experiment 2')
plt.xlabel('Cluster')
plt.ylabel('Number of samples')

plt.subplot(1, 3, 3)
plt.hist(labels3, color='salmon')
plt.title('Experiment 3')
plt.xlabel('Cluster')
plt.ylabel('Number of samples')

plt.tight_layout()
plt.show()
