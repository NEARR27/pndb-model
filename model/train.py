import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from imblearn.over_sampling import SMOTE
import pathlib



# 1. Preparación del dataset

df = pd.read_csv(pathlib.Path('data/pndb.csv'))

y = df.pop('PNDM') 
X = df

# Conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceamos el conjunto de entrenamiento usando SMOTE
print('Balancing the dataset with SMOTE...')
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print('Training model...')
clf = RandomForestClassifier(n_estimators=10000, max_depth=100, random_state=42)
clf.fit(X_train_smote, y_train_smote)

# Evaluamos el modelo en el conjunto de prueba 
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print('Saving model...')
dump(clf, pathlib.Path('model/pndm-prediction-v2.joblib'))

print(y.value_counts())

