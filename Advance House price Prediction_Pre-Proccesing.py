import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\csv files\\ADVANCE_HOUSE.csv")
pd.set_option("display.max_columns",None)     #maked as comment for a while
print(dataset.shape)
print(dataset.head())


#FINDING NULL VALUE


feature_with_null = [x for x in dataset.columns if dataset[x].isnull().sum()>1]
print(feature_with_null)
for x in feature_with_null:
    print(x,np.round(dataset[x].isnull().sum() * 100 / len(dataset[x]),3), '%missing value')



#UNDERSTANDING RELATION BETWEEN NULL AND OUTPUT
data = dataset.copy()

#TRANSFORMING NULL VALUE
data[x] = np.where(data[x].isnull(),1,0)    #CHANGES NULL TO ZERO ELSEWHERE 1


#MEAN SALES PRICE VS NULL VALUES
data.groupby(x)['SalePrice'].mean().plot.bar()
plt.show()                                            #HERE WE HAVE SEEN SALES PRICE IS INCREASED WITH MISSING VALUE ,SO CANNOT DROP NULL FEATURES





#SEPARATING NUMERICAL AND CATEGORICAL FEATURE
numeric_feature = [x for x in dataset.columns if dataset[x].dtype!='O']
print(len(numeric_feature),"are numerical features")



#HANDELING YEAR FEATURE
year_feature = [ x for x in numeric_feature if 'Yr' in x or 'Year' in x]
print(year_feature)
#YEAR VS SALES PRICE
dataset.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year Sold')
plt.ylabel('Median house Price')
plt.title("House Price vs Year Sold")
#plt.show()




#DISTINGUISE BETWEEN DESCRETE AND CONTINUOUS VARIABLE
des_variable = [x for x in numeric_feature if len(dataset[x].unique())<25 and x not in year_feature]
print(len(des_variable))
i=1
for feature in des_variable:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.subplot(8,3,i)
    i=i+1
    plt.xlabel('feature')
    plt.ylabel('sales price')
plt.show()




#continuos variables
cont_features = [x for x in numeric_feature if x not in des_variable+year_feature+['Id']]
print(len(cont_features))
data = dataset.copy()
j=1
for x in cont_features:
    plt.xlabel(x)
    plt.ylabel("count")
    data[x].hist(bins=25)
    plt.subplot(4,4,j)
    j = j + 1

plt.show()
#log transformation to our skewd data
k=1
for x in cont_features:
    data = dataset.copy()
    if 0 in data[x].unique():
        pass
    else:
        data[x] = np.log(data[x])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.subplot(4, 4, k)
        k= k + 1
        data[x].hist(bins=25)                                     #TO UNDERSTAND HOW DATA IS TRANSFORMED
       #plt.scatter(data[x],data['SalePrice'])
        plt.xlabel(x)
        plt.ylabel('SalePrice')
plt.show()






#CHECKING FOR OUTLIERS

l=1
for x in des_variable:
    data = dataset.copy()
    if 0 in data[x].unique():
        pass
    else:
        data[x] = np.log(data[x])
        plt.subplot(4, 4, l)
        l = l + 1
        data.boxplot(column=x)
        #plt.ylabel()
        plt.title(x)
plt.show()


cat_feature = [x for x in dataset.columns if data[x].dtype == 'O']
print(len(cat_feature))
print(dataset[cat_feature].head())

for x in cat_feature:
    n =data[x].nunique(x)
    print("Number of unique elements in " + x,n)  #NUMBER oF UNIQUE ELEMENT IN A COLUMN




#CATEGORICAL FEATURE AND SALES PRICE RELATIONSHIP
m=1
for x in cat_feature:
    data = dataset.copy()
    plt.subplot(11, 4, m)
    m = m + 1
    data.groupby(x)['SalePrice'].median().plot.bar()
    plt.xlabel(x)
    plt.ylabel('salePrice')

plt.show()
