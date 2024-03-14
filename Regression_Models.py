# Objective: Prediction of Life Expectancy

# Preparation of the dataset, apply necessary cleaning, manipulation for ML models.

# Setup
pip install mlxtend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import scipy.stats as stats
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Importing and exploring data
le = pd.read_csv("Expectancy_Data.csv")

# Cleaning Variable names
cols = list(le.columns)
new = []
for i in cols:
    new.append(i.strip().replace('  ', ' ').replace(' ', '_').lower())
le.columns = new
le.columns

# Missing and weird variables detection
le.describe()
le.info()
le.drop(columns='bmi', inplace=True)

# Dealing with missing values
imputed_data = []
for i in list(le.year.unique()):
  year_data = le[le.year == i].copy()
  for j in list(year_data.columns)[3:]:
        year_data[j] = year_data[j].fillna(year_data[j].dropna().mean()).copy()
  imputed_data.append(year_data)

le_2 = pd.concat(imputed_data).copy()

# Dealing with Outliers
cont_vars = list(le_2.columns)[3:]

plt.figure(figsize=(15, 40))
i = 0
for col in cont_vars:
  i += 1
  plt.subplot(9, 4, i)
  plt.boxplot(le_2[col])
  plt.title('{} boxplot'.format(col))
plt.show()

wins_dict = {}

wins_data = winsorize(le_2[cont_vars[0]], limits=(0.006, 0))
wins_dict[cont_vars[0]] = wins_data

wins_data = winsorize(le_2[cont_vars[1]], limits=(0, 0.03))
wins_dict[cont_vars[1]] = wins_data

wins_data = winsorize(le_2[cont_vars[2]], limits=(0, 0.11))
wins_dict[cont_vars[2]] = wins_data

wins_data = winsorize(le_2[cont_vars[3]], limits=(0, 0.002))
wins_dict[cont_vars[3]] = wins_data

wins_data = winsorize(le_2[cont_vars[4]], limits=(0, 0.135))
wins_dict[cont_vars[4]] = wins_data

wins_data = winsorize(le_2[cont_vars[5]], limits=(0.08, 0))
wins_dict[cont_vars[5]] = wins_data

wins_data = winsorize(le_2[cont_vars[6]], limits=(0, 0.185))
wins_dict[cont_vars[6]] = wins_data

wins_data = winsorize(le_2[cont_vars[7]], limits=(0, 0.135))
wins_dict[cont_vars[7]] = wins_data

wins_data = winsorize(le_2[cont_vars[8]], limits=(0.095, 0))
wins_dict[cont_vars[8]] = wins_data

wins_data = winsorize(le_2[cont_vars[9]], limits=(0, 0.018))
wins_dict[cont_vars[9]] = wins_data

wins_data = winsorize(le_2[cont_vars[10]], limits=(0.11, 0))
wins_dict[cont_vars[10]] = wins_data

wins_data = winsorize(le_2[cont_vars[11]], limits=(0, 0.19))
wins_dict[cont_vars[11]] = wins_data

wins_data = winsorize(le_2[cont_vars[12]], limits=(0, 0.11))
wins_dict[cont_vars[12]] = wins_data

wins_data = winsorize(le_2[cont_vars[13]], limits=(0, 0.07))
wins_dict[cont_vars[13]] = wins_data

wins_data = winsorize(le_2[cont_vars[14]], limits=(0, 0.04))
wins_dict[cont_vars[14]] = wins_data

wins_data = winsorize(le_2[cont_vars[15]], limits=(0, 0.035))
wins_dict[cont_vars[15]] = wins_data

wins_data = winsorize(le_2[cont_vars[16]], limits=(0.05, 0))
wins_dict[cont_vars[16]] = wins_data

wins_data = winsorize(le_2[cont_vars[17]], limits=(0.03, 0.005))
wins_dict[cont_vars[17]] = wins_data

# Visualizing features without outliers
plt.figure(figsize=(15,5))
for i, col in enumerate(cont_vars, 1):
    plt.subplot(2, 9, i)
    plt.boxplot(wins_dict[col])
plt.tight_layout()
plt.show()

# Adding the new values without outliers to the dataset
wins_df = le_2.iloc[:, 0:3]
for col in cont_vars:
    wins_df[col] = wins_dict[col]

# Creating a correlation matrix to loof for dependent variables
matrix  = wins_df.corr()
mask = np.triu(np.ones_like(matrix, dtype=bool))

f, ax = plt.subplots(figsize=(25, 15))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(matrix, mask=mask, cmap=cmap, center=0, annot=True,vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Making some visualizations for Exploratory Analysis purposes
sns.lineplot(data=wins_df, x = 'year', y = 'life_expectancy', marker='o')
plt.title('Life Expectancy by Year')
plt.show()

wins_df.groupby('status').life_expectancy.agg(['mean'])

# Linnear Regression Model with One variable

# Variables
X = wins_df[['income_composition_of_resources']]
Y = wins_df[['life_expectancy']]

# Splitting between training and test
x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.2, random_state = 42)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)

# Plotting the predictions for the Regression Model Obtained
plt.scatter(X,Y, color = 'blue')
plt.plot(x_test, y_pred, color = 'red')
plt.xlabel('Income')
plt.ylabel('Life expectancy')

# Checking some metrics to evaluate the model
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE: ', sqrt(mean_squared_error(y_test, y_pred)))
print('R2: ', r2_score(y_test, y_pred))

# Multiple features Regression Model

# Variables
X = wins_df.drop(columns = ['life_expectancy', 'country', 'year', 'status'])
Y = wins_df[['life_expectancy']]

x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.2, random_state = 42)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE: ', sqrt(mean_squared_error(y_test, y_pred)))
print('R2: ', r2_score(y_test, y_pred))

# Using Backward Elimination to check for some improvements on the model
sfs1 = sfs(lin_reg, k_features = 2, forward = False, verbose = 1, scoring = 'neg_mean_squared_error')
sfs1 = sfs1.fit(X,Y)
fin_names = list(sfs1.k_feature_names_)

# Training the model after Backward Elimination 
X = wins_df.drop(columns = ['life_expectancy', 'country', 'year', 'status', 'alcohol', 'infant_deaths', 'percentage_expenditure', 'hepatitis_b',
                            'measles', 'polio', 'total_expenditure', 'gdp', 'population', 'schooling', 'thinness_1-19_years'])

x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.2, random_state = 41)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE: ', sqrt(mean_squared_error(y_test, y_pred)))
print('R2: ', r2_score(y_test, y_pred))
