import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.cluster as skc
import pandas as pd

synNbClasses = 10;

syn=np.empty([2, 0]);
synLabels=[];
for k in range(synNbClasses):
    syn = np.concatenate((syn, np.random.rand(2, 100)+2*k), axis=1);
    synLabels = np.concatenate((synLabels, np.ones(100)*k));


reaNbClasses = 8;

rea1 = np.loadtxt('../data/data1.csv')
rea2 = np.loadtxt('../data/data2.csv')
rea = np.concatenate((rea1, rea2))

reaLabels = rea[:, 0];
rea = rea[:, 1:];




# correspond to (b) 
def node_L2_distances(X, Y):
    #Calculate L2 distances . Note that X and Y are the same matrix.
    distances = np.empty((X.shape[0], Y.shape[0]), dtype='float')
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            distances[i, j] = np.linalg.norm(X[i]-Y[j], ord=2) ** 2
    return distances


# correspond to (a) 
def nearest_neighbor_graph(X, gamma=1):
    #calculate pairwise distances
    A = node_L2_distances(X, X)  
    W = np.exp(-A/(2*gamma**2))
    return W



#use if you wanna set the rbf only for the node's neighborsand 
#the other nodes are set to be 0
#for default, use the above nearest_neighbor_graph(X, gamma=1) instead of this function for part (a).
def nearest_neighbor_graph_(X, gamma=1 ,n_neighbors = 100):
    if n_neighbors > X.shape[0]:
        print('Error: the neighbor node size is invalid...')
    A = node_L2_distances(X, X)  
    sorted_rows_ix_by_dist = np.argsort(A, axis=1)
    #pick up first n_neighbors for each point (i.e. each row)
    nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]
    W = np.zeros(A.shape)
    #for each row, set the entries corresponding to n_neighbors to rbf value
    for row in range(W.shape[0]):
        W[row, nearest_neighbor_index[row]] = np.exp(-W[row, nearest_neighbor_index[row]]/(2*gamma**2))

    #make matrix symmetric by setting edge between two points 
    for r in range(W.shape[0]):
        for c in range(W.shape[0]):
            if(W[r,c] != 0):
                W[c,r] = np.exp(-W[c,r]/(2*gamma**2))
    return W

# correspond to (c) (d)
def compute_laplacian(W):
    #  L = D - W
    d = W.sum(axis=1)
    #create degree matrix
    D = np.diag(d)
    L =  D - W
    return L

# correspond to (e) (f)
def get_eigvecs(L, k):
    #Calculate Eigenvalues and EigenVectors of the Laplacian Matrix.
    eigvals, eigvecs = np.linalg.eig(L)
    #to make sure that eigen vev and values are real value, not complex value which will be error
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    # sort eigenvalues and select k smallest values - get their indices
    ix_sorted_eig = np.argsort(eigvals)[:k]
    #select k eigenvectors corresponding to k-smallest eigenvalues
    return eigvecs[:,ix_sorted_eig]


reaScores_gammas = []
dataset = rea
dataset_labels = reaLabels
k = 10
gammas = [round(i * 0.1, 2) for i in range(1, 21, 1)]
print(gammas)

for gamma in gammas:
    W = nearest_neighbor_graph(dataset, 1/gamma)
    L = compute_laplacian(W)
    E = get_eigvecs(L, k)
    kmeans_result = sk.cluster.KMeans(n_clusters=k).fit(E);
    reaScore = sk.metrics.adjusted_mutual_info_score(dataset_labels, kmeans_result.labels_)
    print('k[{}] gamma[{}]  = my code = {}'.format(k, gamma, reaScore))
    reaScores_gammas.append(reaScore)
    
plt.subplot(1,1,1)
plt.xlabel(" gamma ") 
plt.ylabel("AMI score")
left = np.array(gammas)
height = np.array(reaScores_gammas)
plt.plot(left, height)

