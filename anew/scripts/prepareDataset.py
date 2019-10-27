import pandas as pd
# import the data
hec_data = pd.read_csv("../phl_exoplanet_catalog.csv")
exos = pd.read_csv("../exoplanets.csv")

# remove unconfirmed planets
hec_data = hec_data[hec_data.P_STATUS == 3].drop("P_STATUS", axis="columns")    # P_STATUS - planet status (confirmed = 3)

# leave only the habitable planets
hec_data = hec_data[hec_data.P_HABITABLE != 0]  # P_HABITABLE - planet is potentially habitable index  (1 = conservative, 2 = optimistic)

# join the data with the NASA exoplanets database to create a column that classifies if the
# planet is potentially habitable
habitable_planets_names = hec_data.P_NAME.values
exos["is_habitable"]  = exos["pl_name"].isin(habitable_planets_names)

# columns with more than 40% missing data:
def moreThan40Missing(col):
	numMissing=len(exos[exos[col].isnull()])
	if numMissing/len(exos)>0.4:
		return 1
	return 0
# SignificantMissingData=[x for x in exos.columns if moreThan40Missing(x) ]
exos=exos[exos.isnull()/len(exos)>0.4 ]
# remove columns with more the 40% missing data
exos=exos.drop(SignificantMissingData,axis=1)
print(len(exos.columns))
