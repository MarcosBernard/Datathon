import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Carga de CSVs
train_df = pd.read_csv("./Preprocesamiento/train_preproc.csv",sep=',',encoding='UTF-8')
test_df = pd.read_csv("./Preprocesamiento/test_preproc.csv",sep=',',encoding='UTF-8')

X = train_df.iloc[:,:-1]
Y = train_df['Reached.on.Time_Y.N'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42) 

clf = SVC(kernel='poly', random_state=42,degree=2,C=200)
# Entrenamos
clf.fit(X_train,y_train)
# Predecimos para el dataset de entrenamiento
y_pred = clf.predict(X_test)
# Vemos la sensibilidad o recall.
print('recall:',recall_score(y_test,y_pred))
# Vemos la precisión.
print('accuracy:',accuracy_score(y_test,y_pred))
# Predecimos para la data test
entrenamiento_test = test_df.iloc[:,:]
y_pred_test = clf.predict(entrenamiento_test) #Predice valores para la subtabla pred_test
#Exportamos
salida = pd.DataFrame(y_pred_test,columns=['pred'])
salida.to_csv('MarcosBernard.csv',index=False)