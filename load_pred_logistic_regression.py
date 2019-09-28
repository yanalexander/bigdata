import joblib as jl

classifier = jl.load('models/logistic_regression_diabetes.joblib')
nova_instancia=[[6,148,72,35,0,33.6,0.627,50]]
print("Com o modelo salvo: ")
print(nova_instancia)
print(classifier.predict(nova_instancia))

