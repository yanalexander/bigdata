import pandas as pd
from sklearn.cluster import KMeans
import pickle

vote = pd.read_csv('files/vote.csv')

#Normalizar dados
vote_ = pd.get_dummies(vote)

#Obter cluster
cluster = KMeans(n_clusters=2).fit(vote_)

#Salvar cluster
filename = 'models/kmeans_vote.sav'
pickle.dump(cluster, open(filename, 'wb'))

print(vote_.columns)
