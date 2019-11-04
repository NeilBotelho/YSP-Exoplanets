library(mice)
library(dplyr)
data<-read.csv("forImputation.csv")
#The forImputation.csv file is all the columns in the original exoplanets csv that have numeric data.
#You can do this using the command mentioned here: https://stackoverflow.com/a/34530065
imputed = mice(data,method='rf',maxit=5)
summary(imputed)
completedData<-complete(imputed,1)
write.csv(completedData,'ImputedNumericCols.csv')
