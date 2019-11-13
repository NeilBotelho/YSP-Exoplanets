import pandas as pd
import random as r


#This sets the random seed for  the program. So if 
#anyone else also runs this program, despite using random functions theyre output will be the same as mine
r.seed(42)


#This is the same csv file that i downloaded from https://exoplanetarchive.ipac.caltech.edu
exos=pd.read_csv("exoplanets.csv")

#This is file contains all the columns in the above file that have numerical data, that were selected and 
#imputed(basically the missing values were smartly filled) using the mice library in R. 
#The code used to make this dataset can be found here https://github.com/NeilBotelho/YSP-Exoplanets/blob/unbiased/Impute.R
imputedNumericCols=pd.read_csv("ImputedNumericCols.csv")

#these are the rowid of all the habitable exoplanets in the
#optimistic and conservative sample of habitable exoplents from the PHL habitable exoplanets catalogue
habitable=[2902, 3132, 2155, 2021, 1147, 2541, 2503, 1845, 2135, 2538, 3233, 2880, 2156, 163, 2097, 1424, 3115, 2883, 986, 2316, 1227, 2441, 2031, 2014, 3606, 2129, 703, 114, 2547, 1137, 128, 2189,151, 152, 153, 1604, 2155, 2223, 2882, 3133, 3606, 3716, 3742,3743, 3744]

#Here i replace all the columns of numeric data in the original dataset with
#the columns from the imputed dataset i made using R
for n in imputedNumericCols.columns:
    exos[n]=imputedNumericCols[n]
    
#These are columns that have no impact on whether or not on whether
#a planet is habitable. So i remove them.
remove=['pl_hostname', 'pl_name', 'ra_str', 'dec_str', 'rowupdate', 'pl_def_reflink', 'pl_disc_reflink', 'pl_pelink', 'pl_edelink', 'pl_publ_date', 'hd_name', 'hip_name', 'st_spstr', 'swasp_id']
exos=exos.drop(remove,axis=1)

#Here i check each column in the dataset to see whether the sum of the number of missing values makes
#up more than 40% of that column. If it does then i remove that column
for n in exos.columns:
    if (exos[n].isna().sum()/len(exos) >0.4):
        exos=exos.drop(n,axis=1)
print(len(exos.columns))   #this prints out the columns that remain 

#Here i list out all the columns that have catgorical data(non numeric) and 
#remove all those columns that have more than 10 unique values.
#So a column having the names of 5 different telescopes that were used to discover an exoplanet will
#remain but a column containing the date of its discovery in words will be removed.
Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]
for n in Cat:
    if(len(exos[n].unique())>10):
        exos=exos.drop(n,axis=1)
Cat=[x for x in exos.columns if x not in exos._get_numeric_data().columns]

#Since the categorical columns also have missing data i fill any missing data in these columns with the string "Missing"
exos[Cat]=exos[Cat].fillna(value="Missing")
for n in Cat:
	exos[n]=pd.Categorical(exos[n])

#I add a new column that indicates whether a planet is habitable or not with a 1 or a 0
exos['habitable']=exos.rowid.isin(habitable).replace(True,1).rename('habitable')


#this is the list of features that we will be exporting to a csv
out=[]

def enoughdata(df):
#This is a list of names of all columns that were in the dataframe df 
	trainCols=list(df.columns)
    
#Since we want to use this list as our training columns, we remove the rowid column( as it doesnt effect if a planet is habitable or not) and 
#the habitable column(as that will be used as our prediction variable)
	trainCols.remove('rowid')
	trainCols.remove('habitable')

#We sort the resulting list and join all the strings in it with commas, so that we get a single string.
#So if i get the columns 'a' and 'b'. The resulting string is 'a,b' 
	trainCols.sort()
	trainCols=",".join(trainCols)

#The purpose of sorting the list before hand is so that we can easily check if that combination of 
#columns has been encountered before
#So here we check whether the current string of features has already been selected. If it has we return 0
	if(trainCols in out):
		return 0

#Here we check how many habitable planets are in the dataframe passed as arguement.
#If theres less than 10 then we return 0. 
	num=0
	for n in habitable:
		if n in df.rowid:
			num+=1
	if num<10:
		return 0
    
#Lastly we check whether there enough data to train on. So if there less than 200 rows of
#data we return 0. Else we return the string of training columns
	if len(data)<200:
		return 0
	return trainCols






MinLen=500
numTests=0
numCols=len(exos.columns)

while numTests<20000:
	features=list(exos.columns)
	features.remove('rowid')
	features.remove('habitable')
#These two features are required later on so we add them to selected features right at the beginning
	SelectedFeats=['rowid','habitable']
#Randomly select the number of features that we will be using
	numFeats=r.randint(3,numCols-3)
	for i in range(numFeats):
#Randomly select a feature from the features list and add it to the Selected features list
		featureNum=r.randint(0,len(features)-1)
		SelectedFeats.append(features[featureNum])
		del features[featureNum]

#Make a copy of the columns in the exos dataset having all columns in the selectedFeatures list 
	data=exos[SelectedFeats].copy()
    
#remove rows that have missing data as these cannot be used to train machine learning models
	data=data.dropna()
	trainCols=enoughdata(data)
#	if there isnt enough data trainCols=0 so the following if statement doesnt get executed
	if trainCols:
		print("Currently running record number ",numTests)
		numTests+=1
		out.append(trainCols)

# Create a csv file everytime numTests is divisible by 5000.
#This ensures that if the program exits at some point we know approximately where it stopped 
#and have some data to work with
		if(numTests%5000==0):
			pd.DataFrame(out).to_csv('../randomFeatureGroupings.csv')
 
