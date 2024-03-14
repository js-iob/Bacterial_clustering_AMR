#Author: K.T. Shreya Parthasarathi
#Script: Clustering microbial species using Affinity propagation algorithm

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import time
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors
import plotly.express as px


t0 = 0
def eta(t=None):
    global t0
    if t is not None:
        t0 = time.time()
        return
    else:
        t1 = time.time()
        t = t1 - t0
        t0 = t1
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        return("Ellapsed time {:0>2}:{:0>2}:{:06.3f}".format(int(hours),int(minutes),seconds))


eta(0)

file = pd.read_csv('final_sequence_kmer_binary_021324.csv', sep=",")
file_t = file.set_index('tenmers').transpose()
print (file_t.shape)


jaccard = 1 - pairwise_distances(file_t.to_numpy(), metric='jaccard')
user_distance = pd.DataFrame(jaccard, columns=file_t.index.values, index=file_t.index.values)
user_distance.to_csv('Jaccard_distance_matrix_021624.csv')
print (user_distance.head())

data = user_distance
cor=data.corr()
fig = plt.figure(figsize=(30,30))
sns.heatmap(cor, square = True, cmap='bwr')
plt.xticks(rotation=90)
plt.savefig('Jaccard_corr_021624.pdf')

scaler = StandardScaler()
X_std = scaler.fit_transform(user_distance)

sklearn_pca = PCA(n_components = 1)
Y_sklearn = sklearn_pca.fit_transform(X_std)
print (Y_sklearn)

print ('#################Affinity Propagation#################')
from sklearn.cluster import AffinityPropagation
AP = AffinityPropagation(damping=0.9, max_iter=1000)
fitted = AP.fit(Y_sklearn)
n_clusters_ = len(fitted.cluster_centers_indices_)
print("Affinity-propagation: Number of Clusters: ",n_clusters_)
#print(fitted)
prediction_AP = AP.fit_predict(Y_sklearn)
data["Cluster"] = prediction_AP

data.to_csv('trial_AP_clustering_output_021624.csv')

# Evaluate clustering using Silhouette coefficient
score = metrics.silhouette_score(Y_sklearn, prediction_AP, metric='euclidean')
print ('Silhouette coefficient = ', score, '\n')

# Evaluate clustering using Calinski-Harabasz index
calinski_harabasz = metrics.calinski_harabasz_score(Y_sklearn, prediction_AP)
print(f'Calinski-Harabasz Index: {calinski_harabasz}')

# Evaluate clustering using Davies-Bouldin index
davies_bouldin = metrics.davies_bouldin_score(Y_sklearn, prediction_AP)
print(f'Davies-Bouldin Index: {davies_bouldin}')

print (eta)

