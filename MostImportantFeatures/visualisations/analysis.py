import pandas as pd
import numpy as np

randFeatGroups=pd.read_csv("../randomFeatureGroupings.csv")
randFeatGroups['index']=randFeatGroups.index

scoresOnly=pd.read_csv("../Scores.csv")
scoresOnly=pd.DataFrame(scoresOnly)
scoresOnly
scoresOnly['index']=scoresOnly.index

merged=scoresOnly.merge(randFeatGroups,on='index')

merged=merged.dropna()
#Sort the data by the score with the highest values at the top
merged=merged.sort_values(by="score",ascending=False)
merged.head(10)




#Get the top 2000 scoring feature groups
top=merged.head(2000)

freq={}
def addToFreq(entry):
    global freq
    items=[x.strip() for x in entry.split(",") if ("err" not in x and "lim" not in x and x!='pl_st_nref' and x!='pl_locale')]
    print(np.array(q.loc[entry]))
    # try:
        # items.remove("pl_locale")
        # items.remove("pl_st_nref")
    # except:
        # pass
    for item in items:
        if item in freq:
            freq[item]+=1
        else:
            freq[item]=1

top.features.apply(lambda x: addToFreq(x))
freq
Frequency=sorted(freq, key=freq.get)
Frequency.reverse()
best=Frequency[:10]
trends={}
def addToTrends(entry):
    global trends
    items=[x.strip() for x in entry.split(",") if ("err" not in x and "lim" not in x and x!='pl_st_nref' and x!='pl_locale')]
    for item in items:
        if item in trends:
            trends[item]+=1
        else:
            trends[item]=1

z=q[q.features.str.contains(best[0])]
trends={}
z.features.apply(lambda x:addToFreq(x))
