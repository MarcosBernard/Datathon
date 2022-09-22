import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

data = pd.read_csv('./Datasets/E-Commerce_test.csv',sep=';',encoding='UTF-8')

campo = 'Customer_care_calls'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Customer_rating'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Cost_of_the_Product'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Discount_offered'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Weight_in_gms'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

#Variables cualitativas
label_encoder = preprocessing.LabelEncoder()

campo = 'Warehouse_block'
data[campo] = label_encoder.fit_transform(data[campo])

campo = 'Mode_of_Shipment'
data[campo] = label_encoder.fit_transform(data[campo])

campo = 'Product_importance'# Ordinal, label_encoding
data[campo] = label_encoder.fit_transform(data[campo])

campo = 'Gender'# Nominal, label_encoding
data[campo] = label_encoder.fit_transform(data[campo])

campo = 'Warehouse_block'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Mode_of_Shipment'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Product_importance'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])

campo = 'Gender'
scaler = StandardScaler()
scaler.fit(data[[campo]])
data[campo] = scaler.transform(data[[campo]])


#Creando copia del dataframe original
datac = data.copy()

datac.drop(columns=['ID','Gender','Mode_of_Shipment','Warehouse_block'],inplace=True)

datac.to_csv("Preprocesamiento/test_preproc.csv",index=False)