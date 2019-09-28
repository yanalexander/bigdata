import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

model_file = 'models/kmeans_vote.sav'

kmeans = pickle.load(open(model_file, 'rb'))

#Criar um módulo de inferência: receber uma nova instância e verificar (predict)
result = kmeans.predict([[1,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1]])
print(result)