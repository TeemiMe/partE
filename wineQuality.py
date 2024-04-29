import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("winequality-white.csv", sep=';')

#the number of data objects belonging to each cla
class_counts = data['quality'].value_counts()
data = data.drop_duplicates()
print("Class Counts:", class_counts)

#number of features
num = data.shape[1]

#names of the features
feature_names = data.columns.tolist()

#value types of each feature
value_types = data.dtypes.tolist()

#range of values for each feature
value_ranges = data.describe().loc[['min', 'max']]

#presenting the information
featureInfo = pd.DataFrame({
    "Feature": feature_names,
    "Value Type": value_types,
    "Range of Values": [f"{min_val} - {max_val}" for min_val, max_val in zip(value_ranges.loc['min'], value_ranges.loc['max'])]
})

print(featureInfo)

#print(data.head())

correlationMatrix = data.corr()

#correlation heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlationMatrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation')
plt.title('Correlation Heatmap of Wine Quality Dataset')
plt.xticks(range(len(correlationMatrix)), correlationMatrix.columns, rotation=45)
plt.yticks(range(len(correlationMatrix)), correlationMatrix.columns)
plt.show()

#print(data.describe())

data.boxplot()
plt.title('Boxplot of Wine Quality Variables')
plt.xticks(rotation=45)
plt.show()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
max = Q1 - 1.5 * IQR
min = Q3 + 1.5 * IQR

outliers = data[((data >= max) & (data <= min)).all(axis=1)]

outliers.boxplot()
plt.title('Boxplot of Wine Quality Variables')
plt.xticks(rotation=45)
plt.show()

data['quality_category'] = np.where(data['quality'] <= 5, 'low', 'high')

cols = data.select_dtypes(include=np.number).columns

# Calculate correlation matrix
corr_matrix = data[cols].corr()

data.drop(columns='quality', inplace=True)

sns.pairplot(data, hue='quality_category', palette={"low": "blue", "high": "red"})
plt.show()

data['quality_category'] = np.where(data['quality'] <= 5, 'low', 'high')