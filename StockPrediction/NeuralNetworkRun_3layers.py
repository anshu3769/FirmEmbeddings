import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from sklearn.model_selection import train_test_split
import random

# Load training and testing data
training_data = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/xtrain_stock.pkl');
validating_data = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/xval_stock.pkl');
testing_data = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/xtest_stock.pkl');

# Target train and test
y_training = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/ytrain_stock.pkl');
y_validating = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/yval_stock.pkl');
y_testing = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/Data_stock_all/ytest_stock.pkl');


#Convert to array
X_train = training_data.values
X_test = testing_data.values
X_val = validating_data.values

y_train = y_training.values
y_test = y_testing.values
y_val = y_validating.values

# Dimensions of dataset
num_of_inputVectors = X_train.shape[0]
num_of_features = X_train.shape[1]

print("Number of data points: ", num_of_inputVectors)
print("Number of features: ", num_of_features)

#Create placeholder for input and output
X_input = tf.placeholder(dtype=tf.float32,shape=[None,492])
y_output = tf.placeholder(dtype=tf.float32,shape=[None,1])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


#Model architecture parameters
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([num_of_features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

#Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

#Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))



# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_3, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
#wordVector_to_columns = tf.cast(wordVector_to_columns,tf.float32)
hidden_1 = tf.nn.tanh(tf.add(tf.matmul(X_input, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.tanh(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.tanh(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
#hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, y_output))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.00001, epsilon=0.00001).minimize(mse)

# Session
net = tf.InteractiveSession()

#Init
net.run(tf.global_variables_initializer())

# Fit neural net
batch_size = 10000
mse_train = []
mse_val = []

shuffle_indices = np.random.permutation(np.arange(len(y_test)))
X_test_part = X_test[shuffle_indices]
y_test_part = y_test[shuffle_indices]
y_test_part = y_test_part[0:20000]
X_test_part = X_test_part[0:20000]
y_test_part = y_test_part.reshape((20000,1))


shuffle_indices = np.random.permutation(np.arange(len(y_val)))
X_val_part = X_val[shuffle_indices]
y_val_part = y_val[shuffle_indices]
y_val_part =y_val_part[0:10000]
X_val_part = X_val_part[0:10000]
y_val_part = y_val_part.reshape((10000,1))


np.savetxt('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/actual.txt', y_test_part, fmt='%f')


# Run
epochs = 10
for e in range(epochs):


    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        batch_y = batch_y.reshape((10000,1))
        
        # Run optimizer with batch
        net.run(opt, feed_dict={X_input: batch_x, y_output: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X_input: batch_x, y_output: batch_y}))
            mse_val.append(net.run(mse, feed_dict={X_input: X_val_part, y_output: y_val_part})) 
            
            print('MSE Train: ', mse_train[-1])
            print('MSE Val: ', mse_val[-1])
            
            
mse_test=net.run(mse, feed_dict={X_input: X_test_part, y_output: y_test_part})
pred1 = net.run(out, feed_dict={X_input: X_test_part})
prediction1 = np.asarray(pred1)
print("Prediction is")
print(pred1)
print("MSE Test - ",mse_test)
np.savetxt('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/prediction.txt', prediction1, fmt='%f')


           
