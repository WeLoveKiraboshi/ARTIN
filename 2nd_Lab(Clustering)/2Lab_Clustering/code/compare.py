import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.cluster as skc
import pandas as pd

reaNbClasses = 8;

rea1 = np.loadtxt('../data/data1.csv')
rea2 = np.loadtxt('../data/data2.csv')
rea = np.concatenate((rea1, rea2))

reaLabels = rea[:, 0];
rea = rea[:, 1:];


dataset = rea
dataset_labels = reaLabels
ks = list(range(20)) 
for i in ks:
    ks[i] += 5


clustering = skc.SpectralClustering(n_clusters=30, affinity='rbf', gamma = 1.5, random_state=1).fit(dataset)
reaScore = sk.metrics.adjusted_mutual_info_score(dataset_labels, clustering.labels_)
res = sk.cluster.KMeans(n_clusters=k).fit(dataset)
reaScore_ = sk.metrics.adjusted_mutual_info_score(dataset_labels, res.labels_)    
print('k[{}] Spectral = {} kmeans = {} '.format(k, reaScore, reaScore_))
exit(0)
    
reaScores = []
reaScores_kmeans = []
print(ks)
for k in ks:
    clustering = skc.SpectralClustering(n_clusters=k, affinity='rbf', gamma = 1.5, random_state=None).fit(dataset)
    reaScore = sk.metrics.adjusted_mutual_info_score(dataset_labels, clustering.labels_)
    reaScores.append(reaScore)
    print('end spectral clustering')
    res = sk.cluster.KMeans(n_clusters=k).fit(dataset)
    reaScore_ = sk.metrics.adjusted_mutual_info_score(dataset_labels, res.labels_)
    reaScores_kmeans.append(reaScore_)
    print('k[{}] Spectral = {} kmeans = {} '.format(k, reaScore, reaScore_))
    
    
plt.subplot(1,1,1)
plt.figure
plt.xlabel(" k ") 
plt.ylabel("AMI score")
left = np.array(ks)
height = np.array(reaScores)
plt.plot(left, height)
height2=np.array(reaScores_kmeans)
plt.plot(left, height2)
plt.show()
