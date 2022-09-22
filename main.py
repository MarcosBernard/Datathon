import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


#Carga de CSVs
train_df = pd.read_csv("./Preprocesamiento/train_preproc.csv",sep=',',encoding='UTF-8')
test_df = pd.read_csv("./Preprocesamiento/test_preproc.csv",sep=',',encoding='UTF-8')

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
print('recall:',recall_score(train_df['Reached.on.Time_Y.N'].values,y_pred))
print('accuracy:',accuracy_score(train_df['Reached.on.Time_Y.N'].values,y_pred))
# Predecimos
entrenamiento_test = test_df.loc[:,['Discount_offered','Weight_in_gms','Prior_purchases']]
y_pred_test = clf.predict(entrenamiento_test.values) #Predice valores para la subtabla pred_test



#Exportamos
salida = pd.DataFrame(y_pred_test,columns=['pred'])
salida.to_csv('MarcosBernard.csv',index=False)

print(salida.value_counts())