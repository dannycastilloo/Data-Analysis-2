import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

data = pd.read_csv('3/aids_clinical_1.csv', delimiter=';')

X = data.drop(columns=['str2'])
y = data['str2']

# Entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Árbol de decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# AUC
y_train_pred = model.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_train_pred)
print("AUC en datos de entrenamiento:", auc_train)

# Matriz de confusión
y_train_pred_binary = model.predict(X_train)
conf_matrix_train = confusion_matrix(y_train, y_train_pred_binary)
print("Matriz de Confusión en datos de entrenamiento:")
print(conf_matrix_train)
