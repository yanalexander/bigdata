import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import joblib as jl

diabetes = pd.read_csv("files/diabetes.csv")

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']

# Divide aleatoriamentes os conjuntos em teste e treino
X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)

# Criar e treinar modelo de regressão
logreg = LogisticRegression(solver='liblinear')
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train)

jl.dump(rfe,'models/logistic_regression_diabetes.joblib')
