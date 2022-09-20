import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

#Carga de CSVs
train_df = pd.read_csv("./Datasets/E-Commerce_train.csv",sep=';',encoding='UTF-8')
test_df = pd.read_csv("./Datasets/E-Commerce_test.csv",sep=';',encoding='UTF-8')

#Escojemos los campos con mayor relevancia para entrenar
#ver Correlación.ipynb
entrenamiento = train_df.loc[:,['Discount_offered','Weight_in_gms','Prior_purchases']]

# Instanciamos un objeto de la clase DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 3, random_state = 42) 

# Entrenamos el modelo
clf.fit(entrenamiento.values,train_df['Reached.on.Time_Y.N'].values)
# Predecimos para el dataset de entrenamiento
y_pred = clf.predict(entrenamiento.values)
# Vemos la sensibilidad o recall.
print(recall_score(train_df['Reached.on.Time_Y.N'].values,y_pred))

# Predecimos
entrenamiento_test = test_df.loc[:,['Discount_offered','Weight_in_gms','Prior_purchases']]
y_pred_test = clf.predict(entrenamiento_test.values) #Predice valores para la subtabla pred_test

#Exportamos
salida = pd.DataFrame(y_pred_test)
salida.to_csv('MarcosBernard.csv',index=False)