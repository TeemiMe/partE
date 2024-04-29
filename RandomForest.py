import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("winequality-white.csv", sep=';')

data = data.drop_duplicates()

data['quality_category'] = np.where(data['quality'] <= 5, 0, 1)

#remove quality column
data.drop(columns='quality', inplace=True)

#separate features and target
X = data.drop(columns='quality_category')
y = data['quality_category']

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

#finding the number of data objects added to the test and training datasets and the information on how many data objects from each class are included in training and test sets
totSamples = len(data)

totTrainSamples = len(X_train)
totTestSamples = len(X_test)

trainPrec = (totTrainSamples / totSamples) * 100
testPrec = (totTestSamples / totSamples) * 100

class_counts = data['quality_category'].value_counts()

trainCC = y_train.value_counts()
testCC = y_test.value_counts()

classPrec = (class_counts / totSamples) * 100
trainClassPrec = (trainCC / totTrainSamples) * 100

testClassPrec = (testCC / totTestSamples) * 100

print("Total number of data objects in the training dataset:", totTrainSamples, f"({trainPrec:.2f}%)")
print("Total number of data objects in the test dataset:", totTestSamples, f"({testPrec:.2f}%)")
print()
print("Class Distribution in the Training Dataset:")
print(trainCC)
print(trainClassPrec)
print()
print("Class Distribution in the Test Dataset:")
print(testCC)
print(testClassPrec)

rfr = RandomForestRegressor(random_state=13)

#exp 1
rfr.fit(X_train, y_train)
y_pred1 = rfr.predict(X_test)
mae1 = mean_absolute_error(y_test, y_pred1)
mse1 = mean_squared_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)

#exp 2 (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
rfr_cv.fit(X_train, y_train)
best_rfr = rfr_cv.best_estimator_
y_pred2 = best_rfr.predict(X_test)
mae2 = mean_absolute_error(y_test, y_pred2)
mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

#exp 3 (Changing other hyperparameters)
rfr_custom = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=13)
rfr_custom.fit(X_train, y_train)
y_pred3 = rfr_custom.predict(X_test)
mae3 = mean_absolute_error(y_test, y_pred3)
mse3 = mean_squared_error(y_test, y_pred3)
r2_3 = r2_score(y_test, y_pred3)

print("Experiment 1:")
print("Mean Absolute Error:", mae1)
print("Mean Squared Error:", mse1)
print("R^2 Score:", r2_1)
print()
print("Experiment 2 (GridSearchCV):")
print("Mean Absolute Error:", mae2)
print("Mean Squared Error:", mse2)
print("R^2 Score:", r2_2)
print("Best parameters:", best_rfr.get_params())
print()
print("Experiment 3 (Custom hyperparameters):")
print("Mean Absolute Error:", mae3)
print("Mean Squared Error:", mse3)
print("R^2 Score:", r2_3)

#mean absolute error for the experiments
mae_values = [mae1, mae2, mae3]

#mean squared error for the experiments
mse_values = [mse1, mse2, mse3]

experiments = ['Experiment 1', 'Experiment 2 (GridSearchCV)', 'Experiment 3 (Custom hyperparameters)']

plt.figure(figsize=(10, 6))

plt.bar(experiments, mae_values, alpha=0.7, label='Mean Absolute Error (MAE)', color='blue')
plt.bar(experiments, mse_values, alpha=0.7, label='Mean Squared Error (MSE)', color='orange')

plt.xlabel('Experiments')
plt.ylabel('Error')
plt.title('Performance Metrics Comparison')
plt.legend()

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()