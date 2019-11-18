from joblib import load
import numpy as np

gbr = load('gbr.joblib')

# 'Year Built', 'Floor Area', 'Acres', '# Bedrooms', '# Bathrooms', '# Half Bathrooms, 'Municipality', 'School Name'
x_new_input = np.array([1993, 1118, 1, 3, 2, 1, 1, 6]).reshape((1, -1))

y_new = gbr.predict(x_new_input)
print(y_new)