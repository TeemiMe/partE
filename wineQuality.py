import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("winequality-white.csv", sep=';')
'''
type(data)

data.shape

nrow_count = data.shape[0]

print(nrow_count)

ncol_count = data.shape[1]

print(ncol_count)

data.columns'''

#data.info()

#list(data)

#data.head()

#data.tail()

#type(data)
#data.shape


data = data.drop_duplicates()
data.dropna(axis=0, how='all')
data.dropna(axis=1, how='all')
data.dropna(axis=0, how='any')
data.dropna(axis=1, how='any')

print(data.describe())
'''
#data.describe()

class_counts = data['quality'].value_counts()
print("Class Counts:", class_counts)


num_features = data.shape[1]

# Get the names of the features
feature_names = data.columns.tolist()

# Get the value types of each feature
value_types = data.dtypes.tolist()

# Get the range of values for each feature
value_ranges = data.describe().loc[['min', 'max']]

# Create a DataFrame to present the information
feature_info = pd.DataFrame({
    "Feature": feature_names,
    "Value Type": value_types,
    "Range of Values": [f"{min_val} - {max_val}" for min_val, max_val in zip(value_ranges.loc['min'], value_ranges.loc['max'])]
})

# Print the feature information table
print(feature_info)
'''