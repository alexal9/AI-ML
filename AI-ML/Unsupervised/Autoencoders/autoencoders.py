# Autoencoders

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# reusing the same dataset in RBM example

# preparing training and test set
training_set = pd.read_csv('../Boltzmann Machines/ml-100k/u1.base', sep = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('../Boltzmann Machines/ml-100k/u1.test', sep = '\t')
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

# creating the architecture of the stacked autoencoder neural network
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        # using 20, 10, 20 nodes for the 3 layers of hidden nodes
        # first full connection (encoding)
        self.fc1 = nn.Linear(num_movies, 20)
        # second hidden layer (encoding)
        self.fc2 = nn.Linear(20, 10)
        # third hidden layer (decoding)
        self.fc3 = nn.Linear(10, 20)
        # output
        self.fc4 = nn.Linear(20, num_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # forward propagate input vector x by applying a linear transformation 
        # and passing it through the activation function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
        
# training the stacked autoencoder
epochs = 200

for epoch in range(epochs):
    train_loss = 0
    counter = 0.
    
    for user_id in range(num_users):
        # add batch dimension to input vector since pytorch expects 2d array
        input_row = Variable(training_tensor[user_id]).unsqueeze(0)
        target = input_row.clone()
        # we are interested in users who rated at least one movie
        if torch.sum(target.data > 0) > 0:
            # forward propagate and get output
            output = sae(input_row)
            # don't compute gradient with respect to the target, only the input
            target.require_grad = False
            # discard values for movies the user didn't rate 
            # so the optimizer only works on movies that the user has rated 
            output[target.data == 0] = 0
            # calculate the MSE loss
            loss = criterion(output, target)
            # obtaining the average error mean to consider movies with nonzero ratings
            # ensure denominator is nonzero 
            # (even though in our if condition we know for sure that sum(target > 0) > 0)
            mean_corrector = num_movies / float(torch.sum(target.data > 0) + 1e-10)
            # figure out direction of update for weights
            loss.backward()
            # obtain training loss from the loss function and adjust with the mean corrector factor
            # since the training loss is the squared error, we take the sqrt of the quantity
            train_loss += np.sqrt(loss.data * mean_corrector)
            counter += 1.
            # figure out how much the weights should be adjusted
            optimizer.step()
            
    print('Epoch:', epoch + 1, "Loss:", train_loss / counter)

# testing the stacked autoencoder
test_loss = 0
counter = 0.

for user_id in range(num_users):
    input_row = Variable(training_tensor[user_id]).unsqueeze(0)
    target = Variable(test_tensor[user_id]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input_row)
        target.require_grad = False
        output[target.data == 0] = 0
        loss = criterion(output, target)
        mean_corrector = num_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        counter += 1
        
print("Test loss:", test_loss / counter)

# looking at predictions vs actuals
input_row = Variable(training_tensor[0]).unsqueeze(0)
output = sae(input_row)
target = Variable(test_tensor[0]).unsqueeze(0)
target.require_grad = False

size = len(target[target > 0])
# movie ids are 1-based
movie_ids = np.array([i+1 for i,val in enumerate(target[0]) if val > 0])

# using numpy to display predictions rounded to nearest integer
# np.set_printoptions(precision = 0)
# print( np.concatenate( (movie_ids, np.reshape(output.data[target > 0], (size, 1)), np.reshape(target[target > 0], (size, 1))), axis = 1) )

# create dataframe to show predictions vs actuals on movies with nonzero ratings
movie_info = pd.read_csv('../Boltzmann Machines/ml-100k/u.item', sep = '|', header = None, engine = 'python', encoding = 'latin-1')
predictions = pd.DataFrame()
predictions['movie_id'] = movie_ids
predictions['movie_name'] = movie_info[movie_info[0].isin(movie_ids)][1].values
predictions['prediction'] = output.data[target > 0]
predictions['actuals'] = target.data[target > 0]
pd.set_option('precision', 0)
print(predictions)


