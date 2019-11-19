import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import ensemble


data = pd.read_csv('./merged.csv')
data.head()

# Drop unneccesary columns
drop_cols = ['Street #', 'Street Name', 'Parcel ID', 'SWIS Code', 'Property Class Code',
             'School Code', 'Exterior Wall Type', 'Building Style', 'Zip Code']
data.drop(columns=drop_cols, inplace=True)


# Remove any outliers where the sale price is less than $10000 as they are not representative of true home values
data = data.drop(data[data['Sale Price'] < 10000].index)

# Remove any homes that are not of the type single family home (where 'Property Class Name' != '1 Family Res')
data = data.drop(data[data['Property Class Name'] != '1 Family Res'].index)
data.drop(columns='Property Class Name', inplace=True)

label_encoder = preprocessing.LabelEncoder()

# Convert the categorical columns into numeric values
data['City'] = label_encoder.fit_transform(data['City'])
data['Municipality'] = label_encoder.fit_transform(data['Municipality'])
data['School Name'] = label_encoder.fit_transform(data['School Name'])

data = data.dropna()

target = data['Sale Price']
data2 = data[['Year Built', 'Floor Area', 'Acres', 'Municipality', 'School Name', '# Bedrooms', '# Bathrooms', '# Half Bathrooms']].copy()

x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.10, random_state=4)

from sklearn import ensemble

learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]  # learning_rate=0.1 (4th)
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500, 600, 700] # n_estimators=100 (8th)
max_depths = np.linspace(1, 32, 32, endpoint=True) # max_depth=4
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) # very little influence (.40, 4th element)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True) # Leave as default or 1 (0.1, 1st element)



print('\nLearning Rates \n')
num = 1
for rate in learning_rates:
    gbr = ensemble.GradientBoostingRegressor(learning_rate=rate)
    gbr.fit(x_train, y_train)

    print(num, ' ', gbr.score(x_test, y_test))
    num += 1


print('\nNumber of estimators \n')
num = 1
for n in n_estimators:
    gbr = ensemble.GradientBoostingRegressor(n_estimators=n)
    gbr.fit(x_train, y_train)

    print(num, ' ', gbr.score(x_test, y_test))
    num += 1


print('\nMax depth \n')
num = 1
for depths in max_depths:
    gbr = ensemble.GradientBoostingRegressor(max_depth=depths)
    gbr.fit(x_train, y_train)

    print(num, ' ', gbr.score(x_test, y_test))
    num += 1


print('\nMin Sample Split \n')
num = 1
for min in min_samples_splits:
    gbr = ensemble.GradientBoostingRegressor(min_samples_split=min)
    gbr.fit(x_train, y_train)

    print(num, ' ', gbr.score(x_test, y_test))
    num += 1


print('\nMin Sample Leafs \n')
num = 1
for leaf in min_samples_leafs:
    gbr = ensemble.GradientBoostingRegressor(min_samples_leaf=leaf)
    gbr.fit(x_train, y_train)

    print(num, ' ', gbr.score(x_test, y_test))
    num += 1