import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
habitableRows=[151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742, 3743, 3744]

def numComps(data):
    habitable=data.row_id.isin(habitableRows).replace(True,1).rename('habitable')
    row_id=data.row_id
    data=data.drop('row_id',axis=1)
    scaledData=pd.DataFrame(StandardScaler().fit_transform(data),columns=data.columns)
    pca = PCA().fit(scaledData)
    numComponents=0
    for n in np.cumsum(pca.explained_variance_ratio_):
        if n<0.980:
            numComponents+=1
    pca=PCA(n_components=numComponents).fit_transform(scaledData)
    
    preprocessed=pd.concat([row_id,pd.DataFrame(pca),habitable],axis=1)
    preprocessed=shuffle(preprocessed,random_state=100).reset_index()
    #Split the processed dataset into non habitable and habitable so that we get a 20% habitable in the final test set
    habitableExoplanets=preprocessed[preprocessed.row_id.isin(habitableRows)]
    nonHabitableExoplanets=preprocessed[~preprocessed.row_id.isin(habitableRows)]
    trainCols=[x for x in habitableExoplanets.columns if x not in ['row_id','habitable']]
    #habitable planets train test split
    hTrainX,hTrainY,hTestX,hTestY=train_test_split(habitableExoplanets[trainCols],habitableExoplanets['habitable'])
    #nonhabitable planet train test split
    trainX, trainY,testX,testY=train_test_split(nonHabitableExoplanets[trainCols],nonHabitableExoplanets['habitable'])
    #joingin the 2 splits
    trainX=pd.concat([trainX,hTrainX])
    trainY=pd.concat([trainY,hTrainY])
    testX=pd.concat([testX,hTestX])
    testY=pd.concat([testY,hTestY])
    return trainX,testX,trainY,testY


a=pd.read_csv('simpleImputedMiceCart')
n=['scikitImputed1','missforestSet','simpleImputedMiceRf','simpleImputedMice']
for i in n:
    b=pd.read_csv(i)
    print(i)
    for j in b.columns:
        if j not in a.columns:
            print(j)
    print('\n\n\n')