import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

importer = SimpleImputer(missing_values=np.nan, strategy='mean')
importer.fit(x[:, 1:3])
x[:, 1:3] = importer.transform(x[:, 1:3])
print(x)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)
