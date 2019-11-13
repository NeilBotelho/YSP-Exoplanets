import pandas as pd
import missingno as mn
import matplotlib.pyplot as plt

# import the data
hec_data = pd.read_csv("../../phl_exoplanet_catalog.csv")
exos=pd.read_csv('../../exoplanets.csv')
imputedData=pd.read_csv('../../ImputedNumericCols.csv')
preprocessed=pd.read_csv("../../PreprocessedDataset.csv")

missing={}
for n in exos.columns:
    missing[n]=exos[n].isnull().sum()/len(exos)
# print(missing['gaia_pmlimmissing
x=[x[0] for x in sorted(missing.items(), key = lambda kv:(kv[1]))]
y=[missing[val]*100 for val in x]
y
fig=plt.figure(figsize=(10,10))
plt.xticks([])
plt.yticks(fontsize=20)
plt.xlabel("Sorted Columns",fontsize=25)
plt.ylabel("Missing Percentage",fontsize=25)
plt.title("Missing Data",fontsize=40)
fig= plt.scatter(x,y,s=25)
fig=plt.plot(['a' for n in x],[100 for n in y])
plt.savefig("missingScatterPlot.jpg")

fig=plt.figure(figsize=(60,60))
fig=mn.matrix(exos,inline=False,sparkline=False)
plt.ylabel("row number",fontsize=40)
plt.xlabel("column",fontsize=40)
plt.savefig("RawExosMatrix.jpg")


fig=plt.figure(figsize=(20,20))
fig=mn.matrix(exos[exos.columns[:int(len(exos.columns)/2)]],inline=False,sparkline=False)
plt.ylabel("row number",fontsize=40)
plt.xlabel("column",fontsize=40)
plt.savefig("RawExosMatrix1.jpg")

fig=plt.figure(figsize=(20,20))
fig=mn.matrix(exos[exos.columns[int(len(exos.columns)/2):]],inline=False,sparkline=False)
plt.ylabel("row number",fontsize=40)
plt.xlabel("column",fontsize=40)
plt.savefig("RawExosMatrix2.jpg")

########################################################
#Recreate preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# remove unconfirmed planets
hec_data = hec_data[hec_data.P_STATUS == 3].drop('P_STATUS', axis="columns")    # P_STATUS - planet status (confirmed = 3)

# leave only the habitable planets
hec_data = hec_data[hec_data.P_HABITABLE != 0]  # P_HABITABLE - planet is potentially habitable index  (1 = conservative, 2 = optimistic)

# join the data with the NASA exoplanets database to create a column that classifies if the planet is potentially habitable
habitable_planets_names = hec_data.P_NAME.values
exos["habitable"]  = exos['pl_name'].isin(habitable_planets_names)


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

#End Preprocessing recreation
###################################################################


fig=plt.figure(figsize=(60,60))
fig=mn.matrix(exos,inline=False,sparkline=False)
plt.ylabel("row number",fontsize=40)
plt.xlabel("column",fontsize=40)
plt.savefig("ExosAfterPreparation.jpg")
