library(mice)
library(dplyr)

#Read in Nasa exoplanets data
df<-read.csv("../exoplanets.csv")

#Define a function that returns 1 if more than 30% data is missing
f<- function(x) { 
    return(sum(is.na(x)) < length(x) * 0.4)
}

#Remove any column with more than 30% missing data
df<-df[, vapply(df, f, logical(1)), drop = F]

#Get only numeric columns
numericCols<-dplyr::select_if(df, is.numeric)

imputedCols = mice(numericCols,method='cart',maxit = 10,seed=40)
completedData<-complete(imputedCols,1)
write.csv(completedData,'../ImputedNumericCols.csv')
