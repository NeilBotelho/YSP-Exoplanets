import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# import the data
hec_data = pd.read_csv("../phl_exoplanet_catalog.csv")  
exos=pd.read_csv('../exoplanets.csv')

# remove unconfirmed planets
hec_data = hec_data[hec_data.P_STATUS == 3].drop('P_STATUS', axis="columns")    # P_STATUS - planet status (confirmed = 3)  

# leave only the habitable planets
hec_data = hec_data[hec_data.P_HABITABLE != 0]  # P_HABITABLE - planet is potentially habitable index  (1 = conservative, 2 = optimistic)

# join the data with the NASA exoplanets database to create a column that classifies if the planet is potentially habitable
habitable_planets_names = hec_data.P_NAME.values
exos["habitable"]  = exos['pl_name'].isin(habitable_planets_names)

#Read in the data we imputed using the MICE library in R
imputedData=pd.read_csv('../ImputedNumericCols.csv')

#Merge the imputed data and the exoplanets dataset
for n in imputedData.columns:
    if n not in exos.columns:
        print(n)
    else:
        exos[n]=imputedData[n]

#These columns may introduce bias in the model so remove them
remove=['pl_letter', 'pl_discmethod', 'pl_nnotes', 'ra_str', 'dec_str', 'rowupdate', 'pl_tsystemref', 'pl_def_reflink', 'pl_disc', 'pl_disc_reflink', 'pl_locale', 'pl_facility', 'pl_telescope', 'pl_instrument', 'pl_status', 'pl_pelink', 'st_nts', 'st_nplc', 'st_nglc', 'st_nrvc', 'st_naxa', 'st_nimg', 'st_nspec', 'st_photn', 'st_colorn', 'pl_hostname', 'pl_name', 'ra_str', 'dec_str', 'rowupdate', 'pl_def_reflink', 'pl_disc_reflink', 'pl_pelink', 'pl_edelink', 'pl_publ_date', 'hd_name', 'hip_name', 'st_spstr', 'swasp_id']

exos=exos.drop(remove,axis=1)

# remove columns with more the 40% missing data
def moreThan40Missing(col):
        numMissing=len(exos[exos[col].isnull()])
        if numMissing/len(exos)>0.4:
                return 1
        return 0
SignificantMissingData=[x for x in exos.columns if moreThan40Missing(x) ]
exos=exos.drop(SignificantMissingData,axis=1)

Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]
for n in Cat:
    if(len(exos[n].unique())>10):
        exos=exos.drop(n,axis=1)
Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]

#Fill missing values with np.nan or "Missing"(if categorical)
for n in exos.columns:
    if(n in Cat):
        exos[n]=pd.Categorical(exos[n])
        exos[n]=exos[n].cat.add_categories("Missing").fillna("Missing")
    else:
        exos[n]=exos[n].fillna(np.nan)
exos=exos.set_index("rowid")

#Now we use a simple imputer to impute the remaining missing data.
#Simple imputers only work on numerical data so we only extract the numerical data from the exoplanets dataset
#Separate the habitable planets data from the dataset and impute the resulting two datasets using median imputation
habitable=exos[exos.habitable==True]
nonHabitable=exos[~exos.habitable==True]

imputer=SimpleImputer(missing_values=np.nan,strategy='median')
habitable[habitable._get_numeric_data().columns]=imputer.fit_transform(habitable[habitable._get_numeric_data().columns])
nonHabitable[nonHabitable._get_numeric_data().columns]=imputer.fit_transform(nonHabitable[nonHabitable._get_numeric_data().columns])


# Join the two datasets
Exos=pd.concat([habitable,nonHabitable])

for n in Exos.columns:
    if(n in exos.columns):
        exos[n]=Exos[n]
    else:
        print(n)
#Scale the data so that it has unit variance
NumericCols=[]
for n in exos._get_numeric_data().columns:
    if not (list(exos[n].unique())==[0,1]):
        NumericCols.append(n)


exos[NumericCols]=StandardScaler().fit_transform(exos[NumericCols])
exos=pd.get_dummies(exos)
exos.to_csv("../PreprocessedDataset.csv")

