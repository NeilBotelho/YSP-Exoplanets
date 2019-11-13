import pandas as pd
import random as r
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
r.seed(42)
# import data
exos=pd.read_csv("exoplanets.csv")
# These are the numerical columns that were imputed with the MICE library in R
imputedNumericCols=pd.read_csv("ImputedNumericCols.csv")

# These are the row ids of the planets known to be habitable
habitable=[2902, 3132, 2155, 2021, 1147, 2541, 2503, 1845, 2135, 2538, 3233, 2880, 2156, 163, 2097, 1424, 3115, 2883, 986, 2316, 1227, 2441, 2031, 2014, 3606, 2129, 703, 114, 2547, 1137, 128, 2189,151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742,3743, 3744]

# Merge the imputedColumns with the exos dataset
for n in imputedNumericCols.columns:
	exos[n]=imputedNumericCols[n]
#Remove the following columns from the dataset
remove=['pl_hostname', 'pl_name', 'ra_str', 'dec_str', 'rowupdate', 'pl_def_reflink', 'pl_disc_reflink', 'pl_pelink', 'pl_edelink', 'pl_publ_date', 'hd_name', 'hip_name', 'st_spstr', 'swasp_id']
nonCat=list(exos._get_numeric_data().columns)
nonCat.remove('rowid')
exos=exos.drop(remove,axis=1)

#Remove columns with more than 40% missing data
for n in exos.columns:
    if (exos[n].isna().sum()/len(exos) >0.4):
        exos=exos.drop(n,axis=1)
print(len(exos.columns), "Columns remain in the dataset")

# Remove the categorical columns with more than 10 unique values
Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]
for n in Cat:
    if(len(exos[n].unique())>10):
        exos=exos.drop(n,axis=1)

#Fill the remaining missing values in categorical values with the string 'missing'
Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]
exos[Cat]=exos[Cat].fillna(value="Missing")
for n in Cat:
	exos[n]=pd.Categorical(exos[n])

# Create the 'habitable' column in the exos dataset
exos['habitable']=exos.rowid.isin(habitable).replace(True,1).rename('habitable')

# Remove the '[' character and the ']' character from column names as it causes issues during training
exos.st_metratio=exos.st_metratio.map(lambda x: x.replace("]",""))
exos.st_metratio=exos.st_metratio.map(lambda x: x.replace("[",""))v
exos.to_csv("../useForFiting.csv")
