import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("winequality-white.csv", sep=';')
data = data.drop_duplicates()

#features for clustering
X = data.iloc[:, :-1] 

#scaleing
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

#K-means algorithm
k_values = [2, 3, 4, 5, 6] 
silhouetteScores = []

for k in k_values:
    #clustering model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Xscaled)

    labels = kmeans.labels_
    
    #silhouette scoring
    silhouetteAVG = silhouette_score(Xscaled, labels)
    silhouetteScores.append(silhouetteAVG)
    
    print(f"For k={k}, Silhouette score: {silhouetteAVG}")

#plotting the silhouette score against k
plt.plot(k_values, silhouetteScores, marker='o')
plt.xlabel('Clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette score vs number of clusters')
plt.show()

