import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns',None)
dataset = pd.read_csv("NEW_ADVANCE_HOUSE.csv")
print(dataset.head())
#FOR FEATURE SELECTION
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

#DEPENDENT AND INDEPENDENT VARIABLE
Y = dataset[['SalePrice']]
X = dataset.drop(['Id','SalePrice'],axis=1)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005,random_state=0))    #select variable based on weight
feature_sel_model.fit(X,Y)
#PRINT(FEATURE_SEL_MODEL.GET_SUPPORT())


#COLLECTING SELECTED FEATURES
selected_feature = X.columns[feature_sel_model.get_support()]

print('no of feature',X.shape[1])
print('no of selected feature',len(selected_feature))
print(selected_feature)
#print(X_train[selected_feature].head())
X = X[selected_feature]






#MODEL_BUILDING







from sklearn.model_selection import train_test_split
# split into train test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=1)





# #LINEAR REGRESSSION
# from sklearn.linear_model import LinearRegression
# reg = LinearRegression()
# reg.fit(X_train, Y_train)
#
#
#
#
#
# #PLOTTING RESIDUAL ERRORS IN TEST DATA
#
# plt.scatter(reg.predict(X_test),
#             reg.predict(X_test) - Y_test,
#             color="blue", s=10,
#             label='Test data')
#
# #PLOTTING LINE FOR ZERO RESIDUAL ERROR
#
# plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
# plt.legend(loc='upper right')
# plt.title("Residual errors")
# plt.show()
# from sklearn.metrics import r2_score
#
# Y_pred = reg.predict(X_test)
# print('r2_score is',r2_score(Y_test,Y_pred))
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# rms = sqrt(mean_squared_error(Y_test,Y_pred))
# print('root_mean_error is',rms)
# from sklearn.model_selection import KFold, cross_val_score
#
# k_folds = KFold(n_splits = 5)
#
# scores = cross_val_score(reg,X_train,np.ravel(Y_train), cv = k_folds)
#
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))        #avrg cv score 87.6










# #DecisionTreeRegressor


# from sklearn.tree import DecisionTreeRegressor
#
# # create a regressor object
# regressor = DecisionTreeRegressor(random_state=0)
#
# # fit the regressor with X and Y data
# regressor.fit(X_train, Y_train)
# from sklearn.metrics import r2_score
# Y_pred = regressor.predict(X_test)
# print('r2_score is',r2_score(Y_test,Y_pred))
# from sklearn.model_selection import KFold, cross_val_score
#
# k_folds = KFold(n_splits = 5)
#
# scores = cross_val_score(regressor,X_train,np.ravel(Y_train), cv = k_folds)
#
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))        #avrg cv score 87.9







##HYPERPARAMETER TUNING
# from sklearn.model_selection import GridSearchCV
# parameters={"splitter":["best","random"],
#             "max_depth" : [1,3,5,7,9,11,12],
#            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
#            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#            "max_features":["auto","log2","sqrt",None],
#            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
# grid_search=GridSearchCV(regressor,param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=3)
# grid_search.fit(X_train,np.ravel(Y_train))
# print( grid_search.best_params_)






# #RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
#CREATE REGRESSOR OBJECT
regressor = RandomForestRegressor(bootstrap=True,max_depth=40,max_features= 7,min_samples_leaf=1,min_samples_split =3,n_estimators=1000)

regressor.fit(X,np.ravel(Y))

Y_pred = regressor.predict(X_test)

print(Y_pred)


#CHECKING ACCURACY


from sklearn.metrics import r2_score
print('r2_score is',r2_score(Y_test,Y_pred))
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test,Y_pred))
print('root_mean_error is',rms)


#CROSS VALIDATION
from sklearn.model_selection import KFold, cross_val_score

k_folds = KFold(n_splits = 5)

scores = cross_val_score(regressor,X_train,np.ravel(Y_train), cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))     # avg cv score 87.9




# #HYPERPARAMETER TUNING
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [20,40,80],
#     'max_features': [3,5,7],
#     'min_samples_leaf': [1,2,3],
#     'min_samples_split': [3,5,8],
#     'n_estimators': [100, 200, 300, 1000]
# }
# grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid,
#                           cv = 3, n_jobs = -1, verbose = 2)
# grid_search.fit(X_train,np.ravel(Y_train))
# print( grid_search.best_params_)
#
#


# #XGBOOST REGRESSOR
#
# from xgboost import XGBRegressor
#
# xgb_r = XGBRegressor()
#
# xgb_r.fit(X_train,Y_train)
#
# Y_pred =xgb_r.predict(X_test)
# from sklearn.metrics import r2_score
# print('r2_score is',r2_score(Y_test,Y_pred))            # avg cv score 86.27

