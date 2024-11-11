import kagglehub

# Download latest version
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")

print("Path to dataset files:", path)

import pandas as pd

#sets a seed so the experiments can be reproduced
import numpy as np
np.random.seed(42)

# Load the CSV file into a DataFrame
file_path = path + '/InSDN_DatasetCSV/Normal_data.csv'
data = pd.read_csv(file_path)

# Pre-process the data by removing socket information
data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID', 'Protocol'])

#transforms the data in a time series
#data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.set_index('Timestamp')

#standardization of features except the label
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

#We use one-hot encoding to convert the labeled string to numerical values. In this model, we consider only binary classification to identify the malicious and normal traffic from input data. Therefore, we are encoding the normal string to a binary value of 0 and respectively, all malicious traffic of 1.
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

# Display the shape of the DataFrame
#print(data.shape)

# Display the data types of the DataFrame
#print(data.dtypes)

# Display the count of each label
#print(data['Label'].value_counts())

# Display the first few rows of the DataFrame
#print(data.head(10))


# Repeat it all for the OVS.csv file
file_path = path + '/InSDN_DatasetCSV/OVS.csv'
data_ovs = pd.read_csv(file_path)
data_ovs = data_ovs.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data_ovs = data_ovs.set_index('Timestamp')
data_ovs[data_ovs.columns[:-1]] = scaler.fit_transform(data_ovs[data_ovs.columns[:-1]])
#reduces the size of the dataset but keeping the same proportion of each label
data_ovs = data_ovs.sample(frac=0.1, random_state=42)
#print(data_ovs['Label'].value_counts())
data_ovs['Label'] = data_ovs['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
#print(data_ovs.shape)
#print(data_ovs.dtypes)
#print(data_ovs['Label'].value_counts())
#print(data_ovs.head(10))

# divides normal data in training and testing
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#print(train_data.shape)
#print(test_data.shape)

# includes the OVS data in the testing data
test_data = pd.concat([test_data, data_ovs])
#print(test_data.shape)
#print(test_data['Label'].value_counts())

# trains a LSTM Autoencoder
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

#removes the label from the training data
train_data = train_data.drop(columns=['Label'])

# The input layer
train_data = np.expand_dims(train_data, axis=2)
inputs = Input(shape=(train_data.shape[1], 1))

# The encoder layer
encoded = LSTM(128, activation='tanh', return_sequences=True)(inputs)
encoded = LSTM(64, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(32, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(16, activation='tanh', return_sequences=False)(encoded)

# The decoder layer
decoded = RepeatVector(train_data.shape[1])(encoded)
decoded = LSTM(16, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(32, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(128, activation='tanh', return_sequences=True)(decoded)

# The output layer
output = LSTM(train_data.shape[2], activation='tanh', return_sequences=True)(decoded)

# The model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Trains the model
epochs = 100
learning_rate = 0.0001
batch = 32
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
history = model.fit(train_data, train_data, epochs=epochs, batch_size=batch, callbacks=callbacks, validation_split=0.1).history

# Plots the training and validation loss
import matplotlib.pyplot as plt
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Detects anomalies in the data
train_pred = model.predict(train_data)
train_loss = np.mean(np.abs(train_pred - train_data), axis=1)

plt.hist(train_loss, bins=50)
plt.xlabel('Train Loss')
plt.ylabel('No of samples')
plt.show()

# Calculates the threshold for anomaly detection    
threshold = np.mean(train_loss) + 3 * np.std(train_loss)
print('Threshold:', threshold)

# Detects anomalies in the test data
test_pred = model.predict(test_data.drop(columns=['Label']))
test_loss = np.mean(np.abs(test_pred - test_data.drop(columns=['Label'])), axis=1)

plt.hist(test_loss, bins=50)
plt.xlabel('Test Loss')
plt.ylabel('No of samples')
plt.show()

# Classifies the data as normal or malicious
test_data['Loss'] = test_loss
test_data['Threshold'] = threshold
test_data['Anomaly'] = test_data['Loss'] > test_data['Threshold']
test_data['Prediction'] = test_data['Label'] == 1

# Displays the count of each label
print(test_data['Label'].value_counts())
print(test_data['Prediction'].value_counts())
print(test_data['Anomaly'].value_counts())

# Displays the first few rows of the DataFrame
print(test_data.head(10))

# Evaluates the model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy:', accuracy_score(test_data['Label'], test_data['Prediction']))
print('Recall:', recall_score(test_data['Label'], test_data['Prediction']))
print('Precision:', precision_score(test_data['Label'], test_data['Prediction']))
print('F1 Score:', f1_score(test_data['Label'], test_data['Prediction']))

# Evaluates the model
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_data['Label'], test_data['Prediction']))

# Evaluates the model
from sklearn.metrics import classification_report
print(classification_report(test_data['Label'], test_data['Prediction']))

# Evaluates the model
from sklearn.metrics import roc_auc_score
print('ROC AUC Score:', roc_auc_score(test_data['Label'], test_data['Prediction']))

# Evaluates the model
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_data['Label'], test_data['Prediction'])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Evaluates the model
from sklearn.metrics import precision_recall_curve
from keras.optimizers import Adam
precision, recall, thresholds = precision_recall_curve(test_data['Label'], test_data['Prediction'])
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Evaluates the model

