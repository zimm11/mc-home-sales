from joblib import load
import numpy as np

gbr = load('gbr.joblib')

# 'Year Built', 'Floor Area', 'Acres', 'Municipality', 'School Name', '# Bedrooms', '# Bathrooms', '# Half Bathrooms'
x_new_input = np.array([1980, 1800, 1, 6, 18, 3, 2, 1]).reshape((1, -1))

y_new = gbr.predict(x_new_input)
print(y_new)