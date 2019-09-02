#Used to figure out which columns to use

import pandas as pd

def findHabitable(exos):
    habitable=['GJ 667 C c', 'GJ 667 C e', 'GJ 667 C f', 'Kepler-1229 b', 'Kepler-1652 b', 'Kepler-186 f', 'Kepler-442 b', 'Kepler-62 f', 'LHS 1140 b', 'Proxima Cen b', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g']
    print('num discovered habitable: ',len(habitable))
    missingHabitable=[]
    allPlanets=exos.pl_name.values
    
    for planet in habitable:
        if planet not in allPlanets:
            missingHabitable.append(n)
    if len(missingHabitable)>0:
        print('num habitbale in exos: ',str(len(habitable)-len(missingHabitable)))
        print('missing are:')
        print(missingHabitable)

    else:
        print('num habitbale in exos: ',str(len(habitable)-len(missingHabitable)))


exos=pd.read_csv('exoplanets.csv')
a=exos
print('Initial num cols: ',len(exos.columns))
print("Dropping columns with more than 50% missing data")
print("Droppping columns that correspond to the limit or error margin of a measurement")
for n in exos.columns:
    if exos[n].isna().sum()/len(exos[n]) > 0.5 or "lim" in n or "err" in n:
        exos=exos.drop(n,axis=1)

numCols=len(exos.columns)
print(numCols," columns remain")



