import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import random


from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.svm import SVC
from sklearn.utils import shuffle






## VERIFY LOCATION AND STORE LABEL NAMES

IMDIR = '101_ObjectCategories/'
labelNamesAll = []

for root, dirnames, filenames in os.walk(IMDIR):
    labelNamesAll.append(dirnames)
    #uncomment to check what is found in this folder
    #for filename in filenames:
        #f = os.path.join(root, filename)
        #if f.endswith(('.png', '.jpg', '.jpeg','.JPG', '.tif', '.gif')):
        #    print(f)

labelNamesAll = labelNamesAll[0]

#The list of all labels/directories is
print(labelNamesAll)


import os

#build DATASET from K categories and (up to) N images from category
K = 5
N = 20
imWidth = 100
imHeight = 100

#selection of label indices
X = np.zeros([K*N,imHeight*imWidth]) #data matrix, one image per row
#Y = np.zeros([K*N,1]) #label indices
Y = -np.ones([K*N,1]) #label indices
labelNames = []

#random.seed(a=42)

labelNames = ['stegosaurus', 'Motorbikes', 'crocodile_head', 'bass', 'airplanes']
filedirs = []
for name in labelNames:
    filedirs.append('101_ObjectCategories/' + name + '/')


globalCount = 0
#for i in range(K): 
for i, filedir in enumerate(filedirs):
    """
    while True:
        lab = random.randint(0,len(labelNamesAll)-1)
        if lab not in labelNames:
            break
    #folders are named after the class label
    filedir = os.path.join(IMDIR,labelNamesAll[lab])
    print(filedir)

    #save the name of the class
    labelNames.append(labelNamesAll[lab])   
    """    

    classCount = 0
    for filename in os.listdir(filedir):
        f = os.path.join(filedir, filename)
        if f.endswith(('.jpg')) and (classCount < N):
            #image = skimage.io.imread(f, as_grey=True)
            image = skimage.io.imread(f, as_gray=True)
            image = skimage.transform.resize(image, [imHeight,imWidth],mode='constant')#,anti_aliasing=True)
            X[globalCount,:] = image.flatten()
            Y[globalCount,:] = i
            globalCount += 1
            classCount += 1

#Remove the unused entries of X and Y
print(globalCount)
X = X[:globalCount,:]
Y = Y[:globalCount,:]

#Check the stored classes
print(labelNames)
print(X.shape)
print(Y.T)


# Split in Train and test set with 80% - 20% rule

Ntrain = np.rint(.8*Y.shape[0]).astype(int)
Ntest = Y.shape[0]-Ntrain
print('Training with', Ntrain , 'training samples and ', Ntest, 'testing samples.')

# Randomize the order of X and Y
X, Y = shuffle(X, Y, random_state=0)


# Split the data and labels into training/testing sets
X_train_ = X[0:Ntrain,:]
Y_train_ = Y[0:Ntrain,:]

VALLEN = int(X_train_.shape[0] * 0.2)
X_train = X_train_[:-VALLEN,:]
Y_train = Y_train_[:-VALLEN,:]
X_val = X_train_[-VALLEN:,:]
Y_val = Y_train_[-VALLEN:,:]
X_test = X[Ntrain:,:]
Y_test = Y[Ntrain:,:]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(Y_train.T)
print(Y_val.T)
print(Y_test.T)



from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score



# Functions to compute the errors between prediction and ground truth 
from statistics import mean
from sklearn.metrics import confusion_matrix

def compute_measures(Y_gt,Y_pred, positiveClass=1): 
    measures = dict()
    Y_len = len(Y_gt)
    eps = 1e-12
    # True positives TP : number of prediction that matches the GT
    TP = sum((Y_gt[i] == positiveClass) and (Y_pred[i]==positiveClass) for i in range(Y_len))
    # True negatives TN
    TN = sum((Y_gt[i] != positiveClass) and (Y_pred[i]!=positiveClass) for i in range(Y_len))
    # False positives FP
    FP = sum((Y_gt[i] != positiveClass) and (Y_pred[i]==positiveClass) for i in range(Y_len))
    # False negatives FN
    FN = sum((Y_gt[i] == positiveClass) and (Y_pred[i]!= positiveClass) for i in range(Y_len))
    print('TP ', TP, 'TN ', TN, 'FP', FP, 'FN', FN, 'Total', TP+TN+FP+FN)
    measures['TP'] = TP
    measures['TN'] = TN
    measures['FP'] = FP
    measures['FN'] = FN
    # Accuracy
    measures['accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    # Precision  
    measures['precision'] = TP/(TP+FP+eps)
    # Specificity
    measures['specificity']= TN / (FP + TN + eps)
    # Recall
    measures['recall'] = TP / (TP + FN + eps)
    # F-measure  dice score
    measures['f1'] = 2 * measures['precision'] * measures['recall'] / (measures['recall'] + measures['precision'])
    # Negative Predictive Value
    measures['npv'] = TN / (FN + TN + eps)
    # False Predictive Value
    measures['fpr'] = FP / (FP + TN + eps)
    return measures

def micro_average(measuresList):
    microAverage = dict()
    eps = 1e-12
    
    TP = sum([dict_['TP'] for dict_ in measuresList])
    TN = sum([dict_['TN'] for dict_ in measuresList])
    FP = sum([dict_['FP'] for dict_ in measuresList])
    FN = sum([dict_['FN'] for dict_ in measuresList])
        
    # Accuracy
    microAverage['accuracy'] = (TP + TN) / (TP + FP + TN + FN)
    
    # Precision
    microAverage['precision'] = TP / (TP + FP + eps)
        
    # Specificity
    microAverage['specificity'] = TN / (FP + TN + eps)
    
    # Recall
    microAverage['recall'] = TP / (TP + FN + eps)
    
    # F-measure
    microAverage['f1'] = 2*microAverage['precision']*microAverage['recall']/(microAverage['recall']+microAverage['precision'])
    
    # Negative Predictive Value
    microAverage['npv'] = TN / (FN + TN + eps)
    
    # False Predictive Value
    microAverage['fpr'] = FP / (FP + TN + eps)
    
        
    print('Accuracy ', microAverage['accuracy'], '\n',
          'Precision', microAverage['precision'], '\n',
          'Recall', microAverage['recall'], '\n',
          'Specificity ', microAverage['specificity'], '\n',
          'F-measure', microAverage['f1'], '\n',
          'NPV', microAverage['npv'],'\n',
          'FPV', microAverage['fpr'],'\n')
    
    return microAverage

def macro_average(measuresList):
    macroAverage = dict()

    # Accuracy
    macroAverage['accuracy'] = np.mean([dict_['accuracy'] for dict_ in measuresList])
    
    # Precision
    macroAverage['precision'] = np.mean([dict_['precision'] for dict_ in measuresList])
        
    # Specificity
    macroAverage['specificity']= np.mean([dict_['specificity'] for dict_ in measuresList])
    
    # Recall
    macroAverage['recall'] = np.mean([dict_['recall'] for dict_ in measuresList])
    
    # F-measure
    macroAverage['f1'] = np.mean([dict_['f1'] for dict_ in measuresList])
    
    # Negative Predictive Value
    macroAverage['npv'] = np.mean([dict_['npv'] for dict_ in measuresList])
    
    # False Predictive Value
    macroAverage['fpr'] = np.mean([dict_['fpr'] for dict_ in measuresList])
    
    print('Accuracy ', macroAverage['accuracy'], '\n',
          'Precision', macroAverage['precision'], '\n',
          'Recall', macroAverage['recall'], '\n',
          'Specificity ', macroAverage['specificity'], '\n',
          'F-measure', macroAverage['f1'], '\n',
          'NPV', macroAverage['npv'],'\n',
          'FPV', macroAverage['fpr'],'\n')
    
    return macroAverage



max_score = 0
grid_params = {SVC():
    {"C": np.logspace(-5, 5, 30), 
     "kernel":["poly"], 
     "degree":[2, 3, 4, 5], 
     "gamma": [0.001, 0.0001]}, 
}
 
import warnings
warnings.filterwarnings('ignore')

for model, param in grid_params.items():
    clf = GridSearchCV( SVC(), param, cv=5, n_jobs=-1)
    clf.fit(X_train, Y_train)
    pred_y = clf.predict(X_val)
    score = f1_score(Y_val, pred_y, average="micro")
    if max_score < score:
        max_score = score
        best_param = clf.best_params_

print('------------validation result ----------------')
print('best param = ', best_param)
clf = SVC(C = best_param['C'], degree=best_param['degree'], gamma=best_param['gamma'], kernel=best_param['kernel'])
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
#visualize_SVM(X_test, Y_test.ravel(), Y_pred)

multiclass_poly = []
for k in range(K):
    multiclass_poly.append(compute_measures(Y_test.ravel(),Y_pred, positiveClass=k))
print('Micro-average')
micro_average(multiclass_poly)


