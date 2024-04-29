from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("winequality-white.csv", sep=';')

data = data.drop_duplicates()

#defining the two classes
data['quality_category'] = np.where(data['quality'] <= 5, 'low', 'high')

#remove quality column
data.drop(columns='quality', inplace=True)

#separate features and target
X = data.drop(columns='quality_category')
y = data['quality_category']

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

#scaling
scaler = StandardScaler()
XtrainScaled = scaler.fit_transform(X_train)
XtestScaled = scaler.transform(X_test)

#train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, **kwargs):
    log_reg = LogisticRegression(**kwargs)
    log_reg.fit(X_train, y_train)
    trainScore = log_reg.score(X_train, y_train)
    testScore = log_reg.score(X_test, y_test)
    return trainScore, testScore

#exp 1
trainScore1, testScore1 = train_and_evaluate(XtrainScaled, XtestScaled, y_train, y_test, random_state=0, C=1, max_iter=4000)

#exp 2 (adjusting C parameter)
trainScore2, testScore2 = train_and_evaluate(XtrainScaled, XtestScaled, y_train, y_test, random_state=0, C=0.1, max_iter=4000)

#exp 3 (changing solver)
trainScore3, testScore3 = train_and_evaluate(XtrainScaled, XtestScaled, y_train, y_test, random_state=0, C=1, max_iter=4000, solver='liblinear')

print("Experiment 1 - Training Score:", trainScore1, "Test Score:", testScore1)
print("Experiment 2 - Training Score:", trainScore2, "Test Score:", testScore2)
print("Experiment 3 - Training Score:", trainScore3, "Test Score:", testScore3)

trainScores = [trainScore1, trainScore2, trainScore3]
testScores = [testScore1, testScore2, testScore3]
experiments = ['Experiment 1', 'Experiment 2', 'Experiment 3']

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(experiments))

plt.bar(index, trainScores, bar_width, label='Training Score')
plt.bar(index + bar_width, testScores, bar_width, label='Test Score')

plt.xlabel('Experiments')
plt.ylabel('Scores')
plt.title('Logistic Regression Performance')
plt.xticks(index + bar_width / 2, experiments)
plt.legend()

plt.tight_layout()
plt.show()

#total number of samples
totSamples = len(X)

#samples in training and test datasets
totTrainSamples = len(X_train)
totTestSamples = len(X_test)

#percentages
trainPrec = (totTrainSamples / totSamples) * 100
testPrec = (totTestSamples / totSamples) * 100

print("Number of samples in training dataset:", totTrainSamples)
print("Number of samples in test dataset:", totTestSamples)
print("Percentage of samples in training dataset: {:.2f}%".format(trainPrec))
print("Percentage of samples in test dataset: {:.2f}%".format(testPrec))

#counting class occurrences and calculating percentage
def count_class_members(y):
    class_counts = y.value_counts()
    totalSamples = len(y)
    classPrec = (class_counts / totalSamples) * 100
    return class_counts, classPrec

#class info in training set
trainCC, trainCPrec = count_class_members(y_train)

#class info in test set
testCC, testCPrec = count_class_members(y_test)

print("Training set class distribution:")
print(trainCC)
print("Percentage:")
print(trainCPrec)

print("\nTest set class distribution:")
print(testCC)
print("Percentage:")
print(testCPrec)