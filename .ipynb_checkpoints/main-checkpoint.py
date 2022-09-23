import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

#Carga de CSVs
train_df = pd.read_csv("./Preprocesamiento/train_preproc.csv",sep=',',encoding='UTF-8')
test_df = pd.read_csv("./Preprocesamiento/test_preproc.csv",sep=',',encoding='UTF-8')

#Escojemos los campos con mayor relevancia para entrenar
#ver Correlación.ipynb
X = train_df.loc[:,['Cost_of_the_Product','Discount_offered']]
Y = train_df.loc[:,['Reached.on.Time_Y.N']]
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42) 
s
X = train_df.loc[:,['Product_importance','Discount_offered','Weight_in_gms']]
Y = train_df['Reached.on.Time_Y.N'].values
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42) 

clf = KMeans(n_clusters=2, random_state=0,algorithm='lloyd')
# Entrenamos
#clf.fit(X_train,y_train)
clf.fit(X)
# Predecimos para el dataset de entrenamiento
#y_pred = clf.predict(X_test)
y_pred = clf.predict(X)
# Vemos la sensibilidad o recall.
#print('recall:',recall_score(y_test,y_pred))
print('recall:',recall_score(Y,y_pred))
# Vemos la precisión.
#print('accuracy:',accuracy_score(y_test,y_pred))
print('accuracy:',accuracy_score(Y,y_pred))
# Predecimos para la data test
entrenamiento_test = test_df.loc[:,['Product_importance','Discount_offered','Weight_in_gms']]
y_pred_test = clf.predict(entrenamiento_test) #Predice valores para la subtabla pred_test
#Exportamos
salida = pd.DataFrame(y_pred_test,columns=['pred'])
salida.to_csv('MarcosBernard.csv',index=False)

print(salida.value_counts())