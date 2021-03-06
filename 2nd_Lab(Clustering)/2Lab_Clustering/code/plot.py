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



gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
reaScores_gammas = [0.4954785803215811,0.6790551752356084,0.682060866233112,0.7545351088930872,0.7540948941613439,0.7544768006597345,0.7542799789870099,0.7535112856020005,0.7533237550194093,0.7519902804626276,0.7477121704464602,0.74838827777694,0.7483882777769401,0.7453967759162888,0.4838229933667546,0.3912792070792688,0.6340124091470448, 0.40517125158243394, 0.032068763426584296 , 0.4704121996243954]

plt.subplot(1,1,1)
plt.xlabel(" gamma ") 
plt.ylabel("AMI score")
left = np.array(gammas)
height = np.array(reaScores_gammas)
plt.plot(left, height)
plt.show()
