import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)
dataset = pd.read_csv("D:\csv files\\ADVANCE_HOUSE.csv")
print(dataset.head())





#MISSING VALUE

null_ind = [x for x in dataset.columns if dataset[x].isnull().sum()>1 and dataset[x].dtype == 'O']
for x in null_ind:
    print(x,np.round(dataset[x].isnull().mean(),4),"% are missing")






#REPLACING NULL VALUES
def rep_cate_null(dataset,x):
    data = dataset.copy()
    data[x] = data[x].fillna('missing')
    return data
dataset = rep_cate_null(dataset,null_ind)
print(dataset[null_ind].isnull().sum())





#MISSING VALUE_NUMERICAL VARIABLE
numeric_null_ind = [x for x in dataset.columns if dataset[x].isnull().sum()>1 and dataset[x].dtype != 'O']
for x in numeric_null_ind:
    print(x,np.round(dataset[x].isnull().mean(),4),"% are missing")
    for x in numeric_null_ind:
        median = dataset[x].median()
        dataset[x+'nan'] = np.where(dataset[x].isnull(),1,0)
        dataset[x].fillna(median,inplace = True)
print(dataset[numeric_null_ind].isnull().sum())
print(dataset.head())
numeric_feature = [x for x in dataset.columns if dataset[x].dtype!='O']
print(dataset[numeric_feature].head())
for x in numeric_feature:
    j=1
    for i in range(0,1460):
        if dataset[x][i]!=0:
            j= j+1
            if j ==1460:
                print(x)






#HANDELING YEAR VARIABLE
for x in ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']:
    dataset[x] = dataset['YrSold']-dataset[x]
#print(dataset.head())



print(dataset.shape)


num_feature = ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']
for x in num_feature:
    dataset[x]=np.log(dataset[x])
print(dataset.head())





#HANDELING RARE CAT FEATURE
cat_feature = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']
print(len(cat_feature))

for x in cat_feature:
    temp = dataset.groupby(x)['SalePrice'].count()/len(dataset)
    temp_df = temp[temp>0.01].index
    dataset[x]= np.where(dataset[x].isin(temp_df),dataset[x],'Rare_var')

print(dataset.head())





#ENCODING(NOT USE LABEL ENCODER AS I WANT TO ASSIGN SENSIBLE WEIGHT T OEACH VARIABLE)

categorical_features=[x for x in dataset.columns if dataset[x].dtype=='O']
for x in categorical_features:
    labels_ordered=dataset.groupby([x])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[x]=dataset[x].map(labels_ordered)
dataset.head(10)




#FEATURE SCALING OF NUMERIC VARIABLE
feature_scale = [x for x in dataset.columns if x not in ['Id','SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(dataset[feature_scale])
NewData = pd.concat([dataset[['Id','SalePrice']].reset_index(drop = True),pd.DataFrame(scalar.transform(dataset[feature_scale]),columns=feature_scale)],axis=1)
print(NewData.head())



NewData.to_csv('NEW_ADVANCE_HOUSE.csv',index=False)








