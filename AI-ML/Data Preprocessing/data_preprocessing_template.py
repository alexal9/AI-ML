# data preprocessing template/steps

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset, last column is usually dependent variable
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 
print(X, y, sep = '\n')

# handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print(X)

# encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# alternatively, we can use pandas to do one hot encoding
# and we can use pandas to preprocess the dataframe before we extract values
# https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
# https://pbpython.com/categorical-encoding.html

# view rows with missing values
dataset[dataset.isnull().any(axis = 1)]

# impute mean
dataset_filled = dataset.fillna(dataset.mean())

# for label encoding, we can convert column to be of type category
dataset['Country'] = dataset['Country'].astype('category')
print(dataset['Country'].cat.categories, dataset['Country'].cat.codes, sep = '\n')
dataset['Country_codes'] = dataset['Country'].cat.codes
# we can also use a dictionary for a custom mapping
dataset['Country_codes_custom'] = dataset['Country'].replace({'France': -1, 'Spain': 2, 'Germany': 4})
# note can get columns by index 
print(dataset.iloc[:, [0, -2, -1, 1, 2]])

# one hot encoding and dropping first column to avoid multicollinearity 
dataset_with_dummies = pd.get_dummies(dataset_filled, drop_first = True)
print(dataset_with_dummies)

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train, X_test, sep = '\n')

