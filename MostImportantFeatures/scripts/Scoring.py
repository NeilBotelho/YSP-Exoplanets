import pandas as pd
import numpy as np
# from imblearn import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import random as r
import sys
r.seed(42)
smote = SMOTE(ratio='minority')
exos=pd.read_csv("../useForFitting.csv")
habitableRows=[2902, 3132, 2155, 2021, 1147, 2541, 2503, 1845, 2135, 2538, 3233, 2880, 2156, 163, 2097, 1424, 3115, 2883, 986, 2316, 1227, 2441, 2031, 2014, 3606, 2129, 703, 114, 2547, 1137, 128, 2189,151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742,3743, 3744]
def Preprocessing(df):
	noncat=[x for x in df._get_numeric_data().columns]
	cat=[x for x in df.columns if x not in noncat]
	df=pd.get_dummies(df,columns=cat)
	trainCols=[x for x in df.columns if x not in ['habitable','rowid']]
	validate=[]
	Hcopy=[x for x in habitableRows if x in df.rowid]
	if(len(Hcopy)<13):
		return 0,0,0,0
	numHidden=int(len(Hcopy)/2)
	for i in range(numHidden): 
	    randNum=r.randint(0,len(Hcopy)-1)
	    validate.append(Hcopy[randNum])
	    del Hcopy[randNum]
	
	while len(validate)<100:
	    temp=r.randint(0,3500)
	    if temp not in habitableRows:
	        validate.append(temp)
	validate=df[df.rowid.isin(validate)]
	d=df[~df.rowid.isin(validate.rowid)]
	X=d[trainCols]
	y=d.habitable
	try:
		X, y = smote.fit_sample(X, y)
		validateX,validateY=smote.fit_sample(validate[trainCols],validate.habitable)
	except ValueError:
		return 0,0,0,0		
	return X,y,validateX,validateY

def score(Features):
	feats=Features.split(",")
	feats.append('habitable')
	feats.append('rowid')
	data=[]
	data=exos[feats]
	data=data.dropna()
	if(len(data)<700):
		return np.nan
	X,y,validateX,validateY=Preprocessing(data)
	if (type(X)==type(0)):
		return np.nan
	trainX,testX,trainY,testY=train_test_split(X,y)
	max_depth=[5,7]
	learning_rate=[1,0.1,0.001]
	n_estimators=[10,50,100,150]
	booster=["gbtree","dart"]
	best=-1
	for l in learning_rate:
	  for N in n_estimators:
	    for b in [0,1]:
	      for m in max_depth:
	          testXGB=XGBClassifier(verbosity=0,max_depth=m,learning_rate=l,booster=booster[int(b)],n_estimators=N)
	          testXGB.fit(trainX,trainY,eval_set=[(testX,testY)],early_stopping_rounds=10,verbose=False)
	          b=balanced_accuracy_score(validateY,testXGB.predict(validateX))
	          if best<b:
	            best=b
	return best

randomFeatureGroupings=pd.read_csv("../randomFeatureGroupings.csv")
numChecked=0
out=np.empty((0))
for record in randomFeatureGroupings.features:
	print("Running Record number",numChecked)
	out=np.append(out, score(record))
	numChecked+=1
	if(numChecked%100 ==0):
		print("#"*10,"\n")
		print("Printed at",numChecked)
		pd.Series(out).to_csv("../scoring2.csv",header=['score'],index=False)
pd.Series(out).to_csv("../scores.csv",header=['score'],index=False)
