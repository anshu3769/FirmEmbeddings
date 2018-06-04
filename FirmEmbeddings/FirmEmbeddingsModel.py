import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor
import numpy as np
import pandas as pd
import time
import os
import pickle
import string
import torch.utils.data as data_utils
import psutil
from random import shuffle
from sklearn.utils import shuffle as skshuffle
torch.manual_seed(1)


#Train data
X_train = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/xtrain_firm.pkl')
y_train = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/ytrain_firm.pkl')

#Validation data
X_val = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/xval_firm.pkl')
y_val = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/yval_firm.pkl')

#Test data
X_test = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/xtest_firm.pkl')
y_test = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/ytest_firm.pkl')


class Firm_Embedding_Model(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, embedding_dim, num_firms):
        super(Firm_Embedding_Model,self).__init__()
        
        # input is m x D
        self.linear1 = nn.Linear(input_dim,hidden_layer_dim) # D x H 
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_layer_dim,hidden_layer_dim) # H x H
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_layer_dim,embedding_dim)
        
        self.firm_embedding = nn.Linear(embedding_dim,num_firms) # H x F
        # the output is m x F
        
        self.init_weights()

    def forward(self, X):
        out = F.relu(self.linear1(X))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        out = F.relu(self.linear3(out))
        out = self.firm_embedding(out)
        
        # now we have m x F matrix, for m data points, we can do log softmax
        log_prob = F.log_softmax(out,dim=1)
        return log_prob # for each case data, this is probability of which firm has this stock price change
    
    def init_weights(self):
        linear_layers = [self.linear1,self.linear2,self.linear3,self.firm_embedding]
        for layer in linear_layers:
            layer.weight.data.normal_(0.0,0.1)


BATCH_SIZE = 64
train_dataset = data_utils.TensorDataset(FloatTensor(X_train.as_matrix()),LongTensor(y_train.as_matrix()))
train_loader = data_utils.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=False)



INPUT_DIM = 101
HIDDEN_DIM = 50
EMBED_DIM = 50
NUMBER_FIRMS = 709
model = Firm_Embedding_Model(input_dim=INPUT_DIM,hidden_layer_dim=HIDDEN_DIM,
                        embedding_dim=EMBED_DIM,num_firms=NUMBER_FIRMS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)


N_EPOCH = 13
TRAIN_SIZE = train_dataset.data_tensor.shape[0]

print("Training data size",TRAIN_SIZE)
train_losses = []
val_losses = []

X_val_var = Variable(FloatTensor(X_val.as_matrix()))
y_val_var = Variable(LongTensor(y_val.as_matrix()))
model.eval()
y_pred_val = model.forward(X_val_var)
val_loss = criterion(y_pred_val,y_val_var)
print("initial val loss",val_loss.data[0])
startTime = time.time()
model.train()

for i_epoch in range(N_EPOCH):
    epoch_train_loss = 0
    num_batches_per_epoch = int(TRAIN_SIZE/BATCH_SIZE)
    for i_batch,(X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        X_var, y_var = Variable(X_batch),Variable(y_batch)
        
        y_pred = model.forward(X_var)
        loss = criterion(y_pred,y_var)
        loss.backward()
        
        optimizer.step()
        epoch_train_loss += loss.data[0]
        
    # after each epoch
    
    X_val_var = Variable(FloatTensor(X_val.as_matrix()))
    y_val_var = Variable(LongTensor(y_val.as_matrix()))
    model.eval()
    y_pred_val = model.forward(X_val_var)
    val_loss = criterion(y_pred_val,y_val_var)
    ave_train_loss = epoch_train_loss/num_batches_per_epoch
    print("epoch",i_epoch,"ave_train_loss",
          ave_train_loss,"validation loss:",val_loss.data[0],time.time()-startTime)
    val_losses.append(val_loss.data[0])
    train_losses.append(ave_train_loss)
    model.train()
    
trained_emb = model.firm_embedding.weight.data.numpy() 
print((trained_emb))   
pickle.dump(trained_emb,open("/data/WorkData/firmEmbeddings/Models/FirmEmbeddings/trainedModel.pkl","wb"))
