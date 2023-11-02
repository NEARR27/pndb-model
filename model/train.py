import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import io
import pathlib

# 1. Preparación del dataset

df = pd.read_csv(pathlib.Path('data/pndb.csv'))

y = df.pop('PNDM')  # PNDM es nuestra columna objetivo
X = df

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training model...')
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluamos el modelo en el conjunto de prueba 
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print('Saving model...')
dump(clf, pathlib.Path('model/pndm-prediction-v2.joblib'))