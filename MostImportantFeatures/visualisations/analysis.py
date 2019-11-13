import pandas as pd
import numpy as np
AllCombos=pd.read_csv("../40Missing.csv")
AllCombos['index']=AllCombos.index
mini=pd.read_csv("../scoring2.csv")
mini=pd.DataFrame(mini)
mini['index']=mini.index
q=mini.merge(AllCombos,on='index')
q['score']=q["score_x"]
q=q.drop(["score_y","score_x","index"],axis=1)
q["score"]=q.score.apply(lambda x: x*100)
q=q.dropna()
q=q.sort_values(by="score",ascending=False)
q.head(10)
top=q.head(2000)
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
