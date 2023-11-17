import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

from ucimlrepo import fetch_ucirepo
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

selected_features = ['Alcohol', 'Alcalinity_of_ash', 'Nonflavanoid_phenols']
X_selected = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Matriz de Confusión:')
print(conf_matrix)

with open('resultados.pdf', 'wb') as pdf_file:
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, f'Accuracy: {accuracy}')
    c.drawString(100, 780, 'Matriz de Confusión:')
    c.drawString(100, 760, f'{conf_matrix}')
    c.save()

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.show()