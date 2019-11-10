import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import numpy as np
from joblib import dump


data = pd.read_csv('./merged.csv')
data.head()

# Drop unneccesary columns
drop_cols = ['Street #', 'Street Name', 'Parcel ID', 'SWIS Code', 'Property Class Code',
             'School Code', 'Exterior Wall Type', 'Zip Code']
data.drop(columns=drop_cols, inplace=True)

# Remove any outliers where the sale price is less than $10000 as they are not representative of true home values
data = data.drop(data[data['Sale Price'] < 10000].index)

# Remove any homes that are not of the type single family home (where 'Property Class Name' != '1 Family Res')
data = data.drop(data[data['Property Class Name'] != '1 Family Res'].index)
data.drop(columns='Property Class Name', inplace=True)

# Convert the categorical columns into numeric values
label_encoder = preprocessing.LabelEncoder()
data['Municipality'] = label_encoder.fit_transform(data['Municipality'])
data['School Name'] = label_encoder.fit_transform(data['School Name'])

# Drop any rows with missing NaN values
data = data.dropna()

# Set the x values and y value (dependent variable)
target = data['Sale Price']
data2 = data[['Year Built', 'Floor Area', 'Acres', 'Municipality', 'School Name', '# Bedrooms', '# Bathrooms', '# Half Bathrooms']].copy()

# Split the data into training and testing values
x_train, x_test, y_train, y_test = train_test_split(data2, target, test_size=0.10, random_state=2)

# Utilize the GBR model on the dataset, most accurate model pur Jupyter testing
gbr = ensemble.GradientBoostingRegressor()
gbr.fit(x_train, y_train)

# Print the accuracy of the model
print(round(gbr.score(x_test, y_test) * 100), '%')

# Use joblib library to save trained model for later use
dump(gbr, 'gbr.joblib')