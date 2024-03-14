# Initial Setup
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics

# DATA
loan_1 = pd.read_csv("loan_train.csv")
loan_2 = pd.read_csv("loan_test.csv")

# Cleaning process
mode_value = loan_1['Gender'].mode()[0]
loan_1['Gender'].fillna(mode_value, inplace=True)
loan_1['Gender'] = loan_1['Gender'].replace({'Male':1, 'Female':0})
mode_value_3 = loan_1['Married'].mode()[0]
loan_1['Married'].fillna(mode_value_3, inplace=True)
loan_1['Married'] = loan_1['Married'].replace({'Yes':1, 'No':0})
loan_1['Education'] = loan_1['Education'].replace({'Graduate':1, 'Not Graduate':0})
mode_value_4 = loan_1['Self_Employed'].mode()[0]
loan_1['Self_Employed'].fillna(mode_value_4, inplace=True)
loan_1['Self_Employed'] = loan_1['Self_Employed'].replace({'Yes':1, 'No':0})
loan_1['Area'] = loan_1['Area'].replace({'Semiurban':1, 'Urban':2, 'Rural':3})
mode_value_2 = loan_1['Dependents'].mode()[0]
loan_1['Dependents'].fillna(mode_value_2, inplace=True)
loan_1['Dependents'] = loan_1['Dependents'].replace({'0':1, '1':2, '2':3, '3+':4})
loan_1['Status'] = loan_1['Status'].replace({'Y':1, 'N':0})
loan_1['Term'] = loan_1['Term'].fillna(loan_1['Term'].mean())
mode_value_5 = loan_1['Credit_History'].mode()[0]
loan_1['Credit_History'].fillna(mode_value_5, inplace=True)

# Exploring Data
loan_1.info()
loan_1.dtypes

# Decision Tree Model
X = loan_1.drop(columns =['Status']).values
Y = loan_1['Status'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

dt = DecisionTreeClassifier(criterion='entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)

# f1_score for training data
print(f1_score(Y_train,dt_pred_train))

# f1_score for test_data
dt_pred_test = dt.predict(X_test)
print(f1_score(Y_test, dt_pred_test))

# Random Forest Model
rfc = RandomForestClassifier(criterion='entropy', random_state = 42)
rfc.fit(X_train, Y_train)

# f1_score for test_data
rfc_pred_train = rfc.predict(X_train)
print(f1_score(Y_train, rfc_pred_train))

# f1_score for test_data
rfc_pred_test = rfc.predict(X_test)
print(f1_score(Y_test, rfc_pred_test))
