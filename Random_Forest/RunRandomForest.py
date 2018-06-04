import matplotlib as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


#Load the stock data
stockData = pd.read_csv('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingRandomForest/stockData07to13_logdiff_5_0.1.csv')


#Display  stock data
columns = ['Date','CompanyName','ClosingPrice','City','State','NAICS','LogDiff']
stockData.columns = columns
stockData['Date'] =  pd.to_datetime(stockData['Date'])
print("stockData")
print(stockData)


#Load the case data
caseData = pd.read_pickle('/data/WorkData/firmEmbeddings/Models/StockPredictionUsingRandomForest/ngramdata.pkl')
caseData['Date'] =  pd.to_datetime(caseData['Date'])
print("caseData")
print(caseData)


#Join stock and case data on Date field
joinedData = pd.merge(caseData,stockData,how='inner',on='Date')
printt("joinedData")
printt(joinedData)

#Keep relevant columns in the joined data
joinedData = joinedData.drop(columns=['City','State','NAICS','Month','Date'])
joinedData = joinedData.drop(columns=['ClosingPrice'])



#One hot encode firm names
data_encoded = pd.get_dummies(joinedData)


#Divide the data in train and test set
data_encoded_train = data_encoded[:30000]
data_encoded_test = data_encoded[70001:]


print("Train data count")
print(data_encoded_train.count)

print("Test data count")
print(data_encoded_test.count)


#Get the labels 
labels_train = np.array(data_encoded_train['LogDiff'])
labels_test = np.array(data_encoded_test['LogDiff'])

train_data = np.array(data_encoded_train.drop(columns=['LogDiff']))
test_data = np.array(data_encoded_test.drop(columns=['LogDiff']))


#Initialize Ransdom Forest and fit on data
#Instantiate model with 50 decision trees and 20 leaves
rf = RandomForestRegressor(n_estimators = 50, random_state = 20)

# Train the model on training data
rf.fit(train_data, labels_train);


#Predict on test data and report error
predictions = rf.predict(test_data)
errors = np.zeros(400)

#Calculate the absolute errors
for i in range(400):
    if(labels_test[i] != 0):
        errors[i] = abs(predictions[i] - labels_test[i])/abs(labels_test[i])
    else:    
        errors[i] = abs(predictions[i] - labels_test[i])

#Print out the mean absolute error (mae)
print('Percent Error:', round(np.mean(errors), 2), 'degrees.')


#Scatter plot actual change in stock price ans predicted change in stock price 

print("Scatter plot of actual and predicted stock price change")
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = range(len(predictions))

x = range(1000)
ax1.scatter(x, labels_test[0:1000], s=7, c='b', marker="s", label='actual change in stock')
ax1.scatter(x,predictions[0:1000], s=7, c='r', marker="o", label='predicted change in stock')

plt.legend(loc='upper left');
plt.ylim(-3,3)
plt.show()

