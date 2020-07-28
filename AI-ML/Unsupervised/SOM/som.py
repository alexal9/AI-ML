#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# training SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map())
colorbar()
# largest distances means those nodes are outliers

markers = ['o', 's'] # circle, square
colors = ['r', 'g'] # red, green
for i, x in enumerate(X):
    w = som.winner(x)
    # plot marker with color at center of cell
    plot(w[0] + 0.5, 
         w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# finding the frauds
mappings = som.win_map(X)
distance_map = som.distance_map()
fraud_cells = {(i,j): distance_map[i,j] for i in range(len(distance_map)) for j in range(len(distance_map[0])) if distance_map[i,j] > 0.9}
# frauds = []
# for cell in fraud_cells:
    # frauds += mappings[cell]
    
# convert mappings[cell] to numpy array and reshape to 2D array to properly concatenate
# since some cells will only have one customer, and others will have a list of customers
frauds = np.concatenate( [np.array(mappings[cell]).reshape(-1, 15) for cell in fraud_cells], axis = 0 )
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(i)

# map ids to target value
id_map = dict(zip(dataset.iloc[:, 0].values, y))

print('Fraud Customers IDs (Accepted)')
for i in frauds[:, 0]:
    if id_map[int(i)] == 1:
        print(i)