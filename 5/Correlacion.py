import pandas as pd

data = pd.read_csv('5/aids_clinical_1.csv', delimiter=';')

# Correlación entre 'preanti' y 'wtkg'
correlation = data[['preanti', 'wtkg']].corr()

print(correlation)