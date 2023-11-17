import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

data = pd.read_csv('4/glioma_grading_1.csv', delimiter=';')

X = data.drop(columns=['Grade'])
y = data['Grade']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

columns_train = X_train.columns
X_test = X_test.reindex(columns=columns_train, fill_value=0)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

auc_rf = roc_auc_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Resultados para Random Forest:")
print("AUC:", auc_rf)
print("Accuracy:", accuracy_rf)
