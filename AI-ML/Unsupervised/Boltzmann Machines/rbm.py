# Restricted Boltzmann Machines

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# data from:
# https://grouplens.org/datasets/movielens/100k/
# https://grouplens.org/datasets/movielens/1m/

# exploring the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# preparing training and test set
training_set = pd.read_csv('ml-100k/u1.base', sep = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', sep = '\t')
test_set = np.array(test_set, dtype = 'int')

# getting the number of users and movies by getting max user_id and movie_id
num_users = int( max( max(training_set[:,0]), max(test_set[:,0]) ) )
num_movies = int( max( max(training_set[:,1]), max(test_set[:,1]) ) )

# convert dataset to 2D array with users as rows and movies as columns
def convert(data):
    result = np.zeros((num_users, num_movies), dtype = 'int')
    # last column refers to timestamp and we don't need that information
    for user_id, movie_id, rating, _ in data:
        result[user_id - 1, movie_id - 1] = rating
    # easier to build up numpy array and then convert to list of lists for pytorch
    return result.tolist()
        
converted_training_set = convert(training_set)
converted_test_set = convert(test_set)

# convert the data into Torch tensors
training_tensor = torch.FloatTensor(converted_training_set)
test_tensor = torch.FloatTensor(converted_test_set)

# convert the ratings to binary ratings: 1 (liked), 0 (disliked)
training_tensor[training_tensor == 0] = -1 # replace all zeroes with -1
training_tensor[training_tensor == 1] = 0
training_tensor[training_tensor == 2] = 0 # ratings of 1 and 2 -> disliked
training_tensor[training_tensor >= 3] = 1 # ratings of 3 or more -> liked

test_tensor[test_tensor == 0] = -1
test_tensor[test_tensor == 1] = 0
test_tensor[test_tensor == 2] = 0
test_tensor[test_tensor >= 3] = 1

# creating the architecture of the neural network
class RBM:
    def __init__(self, num_visible, num_hidden):
        self.W = torch.randn(num_hidden, num_visible) # initialize weights
        self.a = torch.randn(1, num_hidden) # initialize biases for hidden nodes
        self.b = torch.randn(1, num_visible) # initialize biases for visible nodes
    
    def sample_hidden(self, x):
        # Gibbs sampling using probability hidden nodes will be activated given visible nodes P(H|V)
        # x: visible nodes
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        prob = torch.sigmoid(activation)
        return prob, torch.bernoulli(prob) # return probabilities and the samples
    
    def sample_visible(self, y):
        # probability visible node will be activated given hidden node P(V|H)
        # y: hidden nodes
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        prob = torch.sigmoid(activation)
        return prob, torch.bernoulli(prob) # return probabilities and the samples
        
    def train(self, v0, vk, ph0, phk):
        # k-step contrastive divergence 
        # v0:  input vector containing ratings of movies from one user
        # vk:  visible nodes obtained after k samplings
        # ph0: vector of probabilities P(H = 1|v0) at the first iteration
        # phk: vector of probabilities P(H = 1|vk) after k samplings 
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

num_visible = len(training_tensor[0]) # number of features in training set 
num_hidden = 100 # number of features we want to detect
batch_size = 32

rbm = RBM(num_visible, num_hidden)

# training the RBM
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    counter = 0.
    
    # getting batches of users
    for user_id in range(0, num_users - batch_size, batch_size):
        vk = training_tensor[user_id: user_id + batch_size] # input batch
        v0 = training_tensor[user_id: user_id + batch_size]
        ph0, _ = rbm.sample_hidden(v0)
        
        # k-steps of contrastive divergence
        for k in range(10):
            _, hk = rbm.sample_hidden(vk) # sample hidden nodes
            _, vk = rbm.sample_visible(hk) # update visible nodes
            vk[v0 < 0] = v0[v0 < 0] # don't update missing ratings
        
        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        counter += 1.
    
    print('Epoch:', epoch + 1, 'Loss:', train_loss/counter)

# testing the RBM
test_loss = 0
counter = 0.
for user_id in range(num_users):
    v = training_tensor[user_id]
    vt = test_tensor[user_id]
    # only need to take one step for predictions
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_hidden(v)
        _, v = rbm.sample_visible(h)
    
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        counter += 1.
    
print('Test loss:', test_loss)





