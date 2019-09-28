import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

model_file = 'models/kmeans1.sav'

kmeans = pickle.load(open(model_file, 'rb'))

result = kmeans.predict([[1,3]])
print(result)