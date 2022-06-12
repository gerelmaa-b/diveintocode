from xml.etree.ElementInclude import include
#import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

print('Start')

########## Problem 1 #############
# Acquisition of data set
data = pd.read_csv('train.csv')
print(data.head())

######### Problem 2, 3 #############
# Investigation of the data set itself
'''
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access 
...
'''
print(data.dtypes)
print(data.select_dtypes(include=object))
print(data.describe())

######### Problem 4 #############
# Dealing with missing values

missing_val = msno.bar(data)
plt.show()

missing_dat = data.isnull().sum()
counted = data.count()
missing_ratio = counted/len(data)

ratio = {'Total':counted, 'missing_ratio' : missing_ratio}
print(pd.DataFrame(ratio))


# delete features
for col in data.columns:
    if data[col].isnull().sum() > 5:
        data.drop(col, axis = 1, inplace = True)

print("The shape of cleaned data is {}".format(data.shape))
# delete samples

data.dropna(axis=0,  inplace = True)
print("Samples of cleaned data is {}".format(len(data)))
######### Problem 5 #############
# Terminology survey
'''
Kurtosis describes the peakedness of the distribution
· Along with skewness, kurtosis is an important descriptive statistic of data distribution. 
However, the two concepts must not be confused with each other. 
Skewness essentially measures the symmetry of the distribution, 
while kurtosis determines the heaviness of the distribution tails.

· It is the sharpness of the peak of a frequency-distribution curve .It is actually the measure of outliers present in the distribution.

· High kurtosis in a data set is an indicator that data has heavy outliers.

· Low kurtosis in a data set is an indicator that data has lack of outliers.

'''

kurtosis = data.kurtosis(axis=0)
print('kurtosis: {}'.format(kurtosis))

'''
Skewness is a measure of asymmetry of a distribution.
In a normal distribution, the mean divides the curve symmetrically into two equal parts at the median and the value of skewness is zero.

· When a distribution is asymmetrical the tail of the distribution is skewed to one side-to the right or to the left.

· When the value of the skewness is negative, the tail of the distribution is longer towards the left hand side of the curve.

· When the value of the skewness is positive, the tail of the distribution is longer towards the right hand side of the curve
'''

skewness = data.skew(axis = 0)
print('skewness of data: {}'.format(skewness))

######### Problem 6 #############
# Confirmation of distribution
kurt_lotarea = data['LotArea'].kurtosis(axis=0)
skew_lotarea = data['LotArea'].skew(axis=0)


sns.displot(data=data['LotArea'],  kde = True)
sns.displot(data=data['LotArea'], log_scale= True, kde = True)
plt.text(50000,100, 'kurtosis = '+str(kurt_lotarea)  , fontsize=10)
plt.text(50000,90, ' skewness = '+ str(skew_lotarea) , fontsize=10)
plt.show()

print('This gives a more clear picture where we can compare the counts even when the value of counts differ a lot from each other.')

'''
If we are using the absolute value of counts, samples with larger values will influence the model more due to their large “magnitude”
If you take log of data, the resulting value will reduce the range of the variable while “preserving” the differences.
'''


######### Problem 7 #############
#  Confirmation of correlation coefficient

corr = data.corr(method = 'pearson')
print(corr)

h = sns.heatmap(corr)
plt.title('Correlation')
plt.show()

#kot = corr.index[corr>=.6]
#sns.heatmap(kot)
#plt.show()

####### data with highly correlated with Sales Price #########
T_corr=corr.index[abs(corr['SalePrice'])>0.51]
g = sns.heatmap(data[T_corr].corr(),annot=True)
plt.show()

print('strong correlations are: GrLivArea_Log and TotRmsAbvGrd; GarageCars and GarageArea; TotalBsmtSF and 1stFlrSF')

print('3 important variables are: OverallQual: Overall material and finish quality, GrLivArea: Above grade (ground) living area square feet and GarageCars: Size of garage in car capacity')
#########################