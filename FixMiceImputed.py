#remove the 773 missing data from pl_rade and imputinghte few missing in gaias and st_k
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
#Removing extra index column added by mice imputation
miceRf=pd.read_csv('dataset/imputedData')
miceRf=miceRf.drop("Unnamed: 0",axis=1)
miceRf['row_id']=miceRf['X']
miceRf=miceRf.drop('X',axis=1)

#Separating the habitable and non habitable exoplanets 
habitable=miceRf[miceRf.row_id.isin([151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742, 3743, 3744])]
miceRf=miceRf[~pd.isnull(miceRf.pl_rade)]
miceRf=miceRf[~miceRf.row_id.isin([151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742, 3743, 3744])]

#I impute  habitable planets and non habitable planets separately
habitable.pl_rade=habitable.pl_rade.fillna(np.nan)
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
habitable=pd.DataFrame(imputer.fit_transform(habitable),columns=habitable.columns)

miceRf=pd.concat([miceRf,habitable])
miceRf=miceRf.fillna(np.nan)
mp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed=mp_mean.fit_transform(miceRf)
miceRf=pd.DataFrame(imputed,columns=miceRf.columns)

for n in miceRf.columns:
    m=miceRf[n].isnull().sum()
    if m>0:
        print(n,str(m))

miceRf.to_csv('dataset/simpleImputedMiceRf',index=False)
