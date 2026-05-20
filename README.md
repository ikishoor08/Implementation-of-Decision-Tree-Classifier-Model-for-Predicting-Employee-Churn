# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import pandas 2.Import Decision tree classifier 3.Fit the data in the model 4.Find the accuracy score


## Program:
```
/*
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHOOR I
RegisterNumber:  212225040190
*/
import pandas as pd

# Load dataset
data = pd.read_csv("Employee.csv")

# Display first rows
print("data.head():")
print(data.head())

# Dataset information
print("\ndata.info():")
print(data.info())

# Null values
print("\nisnull() and sum():")
print(data.isnull().sum())

# Value counts of target column
print("\ndata value counts():")
print(data["left"].value_counts())

# Label Encoding for salary column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

print("\nEncoding Salary Column:")
data["salary"] = le.fit_transform(data["salary"])

print(data.head())

# Feature selection
print("\nx.head():")

x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

print(x.head())

# Target column
y = data["left"]

# Split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

# Train model
dt.fit(x_train, y_train)

# Predictions
y_pred = dt.predict(x_test)

# Accuracy
print("\nAccuracy value:")

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)

# Sample Prediction
print("\nData Prediction:")

prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

print(prediction)

# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)

plt.show() 
*/
```

## Output:
<img width="1232" height="228" alt="image" src="https://github.com/user-attachments/assets/a2c7dfb8-bca9-4e23-a191-d1a4fea5dbec" />
<img width="980" height="366" alt="image" src="https://github.com/user-attachments/assets/26fc5aba-3fde-4d39-a18d-3f721bcb3d3e" />
<img width="843" height="257" alt="image" src="https://github.com/user-attachments/assets/6515c111-0bbd-47ac-879b-480882839c59" />
<img width="501" height="120" alt="image" src="https://github.com/user-attachments/assets/a95464a4-4907-4f25-8860-c1f173604f96" />
<img width="1247" height="223" alt="image" src="https://github.com/user-attachments/assets/f97748d8-c2d9-48eb-ae88-f47658a1d525" />
<img width="1226" height="222" alt="image" src="https://github.com/user-attachments/assets/07808050-6ba0-42e3-a176-e3f805c1015d" />
<img width="452" height="65" alt="image" src="https://github.com/user-attachments/assets/25561201-dd2c-4556-a686-96233703855c" />
<img width="1247" height="133" alt="image" src="https://github.com/user-attachments/assets/65881fc9-6171-4574-ad56-b4903d6887bb" />
<img width="651" height="482" alt="image" src="https://github.com/user-attachments/assets/285fe2f1-62dc-4576-bf26-d9381e33fccd" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
