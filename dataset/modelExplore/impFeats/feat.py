import os
import pandas as pd
import numpy  as np
from sklearn.preprocessing import StandardScaler
dir_path = os.path.dirname(os.path.realpath(__file__))
files=os.listdir(dir_path)
files.remove('featureImportance')
files.remove('feat.py')
files=[dir_path+'/'+x for x in files]
filenames=[x.split("/")[-1] for x in files]
with open(files[0],'r') as f:
		cols=[x.split(" ")[0].strip() for x in f.readlines()]
		f.seek(0)
		scores=[float(x.split(" ")[1].strip()) for x in f.readlines()]
out=pd.DataFrame({'Features':cols,filenames[0]:scores})
out[filenames[0]]=abs(out[filenames[0]])

for n in range(1,len(files)):
		with open(files[n],'r') as f:
			cols=[x.split(" ")[0].strip() for x in f.readlines()]
			f.seek(0)
			scores=[float(x.split(" ")[1].strip()) for x in f.readlines()]
		temp=pd.DataFrame({'Features':cols,filenames[n]:scores})
		out=pd.merge(temp,out,on="Features")
		out[filenames[n]]=abs(out[filenames[n]])

out[filenames]=StandardScaler().fit_transform(out[filenames])
# print(out)

q={}
for n in out['Features']:
	q[n]=0
for n in filenames:
	a=list(out.sort_values(by=n,ascending=True).Features)
	for m in range(len(a)):
		q[a[m]]+=m+int(n[-2:])
a=list(q.keys())

num=0
for n in range(len(a)):
	mx=-1
	mxEle="na"
	for m in a:
		if(q[m]>mx):
			mx=q[m]
			mxEle=m
	a.remove(mxEle)
	num+=1
	print(mxEle)
	if(num==26):
		exit()
