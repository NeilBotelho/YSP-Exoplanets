import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import random as r
import sys
r.seed(42)
smote = SMOTE(ratio='minority')
exos=pd.read_csv("useForRandom.csv")
print("Read exos")
habitableRows=[2902, 3132, 2155, 2021, 1147, 2541, 2503, 1845, 2135, 2538, 3233, 2880, 2156, 163, 2097, 1424, 3115, 2883, 986, 2316, 1227, 2441, 2031, 2014, 3606, 2129, 703, 114, 2547, 1137, 128, 2189,151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742,3743, 3744]
exosColumns=exos.columns


penalty=[None, 'l2' , 'elasticnet']
alpha=[ 0.01, 0.001,0.0001]
max_iter=[100,200]
tol=[0.00001,0.0000001,0.001]
n_iter_no_change=[13,20]
eta0=[1,10,20,70]


def Preprocessing(df):
	noncat=[x for x in df._get_numeric_data().columns]
	cat=[x for x in df.columns if x not in noncat]
	df=pd.get_dummies(df,columns=cat)
	trainCols=[x for x in df.columns if x not in ['habitable','rowid']]
	validate=[]
	Hcopy=[x for x in habitableRows if x in df.rowid]
	if(len(Hcopy)<13):
		# input("less")
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
		# input("Smte")
		return 0,0,0,0
	return X,y,validateX,validateY

def score(Features):
	feats=Features.split(",")
	feats.append('habitable')
	feats.append('rowid')
	data=[]
	try:
		data=exos[feats]
	except KeyError:
		feats=[x for x in feats if x in exosColumns]
		data=exos[feats]
	data=data.dropna()
	if(len(data)<700):
		# print(len(data))
		# input()
		return np.nan
	X,y,validateX,validateY=Preprocessing(data)
	if (type(X)==type(0)):
		# input("Preprocessing")
		return np.nan
	trainX,testX,trainY,testY=train_test_split(X,y)

	best=-1

	for p in penalty:
		for a in alpha:
			for m in max_iter:
				for t in tol:
					for e in eta0:
						for n in n_iter_no_change:
							model= Perceptron(penalty=p,alpha=a,tol=t,n_iter_no_change=n,max_iter=m,n_jobs=-1,early_stopping=True,eta0=e)
							model.fit(trainX,trainY)
							y_preds=model.predict(validateX)
							currScore=balanced_accuracy_score(validateY,y_preds)
							if(currScore>best):
								best=currScore
	return best

try:
	with open("checkpoint",'r') as f:
		checkpoint=int(f.read().strip())
except FileNotFoundError:
	checkpoint=-1
	print("No Checkpoint Found")
placeholder=pd.read_csv("randomFeatureGroupings.csv")
df={'index':[],'score':[]}
for index,record in placeholder.features.iteritems():
	if(index>checkpoint):
		print("Running Record number",index)
		df['index'].append(index)
		df['score'].append(score(record))
		print
		if(index%10==0):
			print("#"*50)
			print("Printed at",index)
			df={'index':[],'score':[]}
			with open("checkpoint",'w') as f:
				f.write(str(index))


pd.DataFrame(data=df).to_csv("intermediateResults.csv",mode="a",header=False,index=False)
pd.DataFrame(data=df).to_csv("Final.csv",header=['index','score'],index=False)
