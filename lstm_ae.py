import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")

# Load and process the first CSV file
file_path = path + '/InSDN_DatasetCSV/Normal_data.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data = data.set_index('Timestamp')
scaler = StandardScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

# Load and process the second CSV file (OVS)
file_path = path + '/InSDN_DatasetCSV/OVS.csv'
data_ovs = pd.read_csv(file_path)
data_ovs = data_ovs.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data_ovs = data_ovs.set_index('Timestamp')
data_ovs[data_ovs.columns[:-1]] = scaler.fit_transform(data_ovs[data_ovs.columns[:-1]])
data_ovs = data_ovs.sample(frac=0.1, random_state=42)
data_ovs['Label'] = data_ovs['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

# Split and combine datasets for training and testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data = pd.concat([test_data, data_ovs])

# Remove label column from training data and reshape for LSTM
train_data_values = np.expand_dims(train_data.drop(columns=['Label']).values, axis=1)

# Define LSTM Autoencoder model for a single timestamp
inputs = Input(shape=(1, train_data_values.shape[2]))
encoded = LSTM(128, activation='tanh', return_sequences=True)(inputs)
encoded = LSTM(64, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(32, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(16, activation='tanh', return_sequences=False)(encoded)

decoded = RepeatVector(1)(encoded)
decoded = LSTM(16, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(32, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(64, activation='tanh', return_sequences=True)(decoded)
decoded = LSTM(128, activation='tanh', return_sequences=True)(decoded)

output = LSTM(train_data_values.shape[2], activation='tanh', return_sequences=True)(decoded)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
epochs = 1#00
learning_rate = 0.0001
batch_size = 32
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
history = model.fit(train_data_values, train_data_values, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.1)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Detect anomalies
train_pred = model.predict(train_data_values)
train_loss = np.mean(np.abs(train_pred - train_data_values), axis=1)

# Calculate anomaly threshold
threshold = np.mean(train_loss) + 3 * np.std(train_loss)
print('Threshold:', threshold)

# Process and predict anomalies in test data
test_data_values = np.expand_dims(test_data.drop(columns=['Label']).values, axis=1)
# Detect anomalies in the test data
test_pred = model.predict(test_data_values)
test_loss = np.mean(np.abs(test_pred - test_data_values), axis=(1, 2))  # Adjusted to calculate per-sample loss

# Add results and calculate performance metrics
test_data['Loss'] = test_loss
test_data['Threshold'] = threshold
test_data['Prediction'] = test_data['Loss'] > test_data['Threshold']

# Display performance metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve

print('Accuracy:', accuracy_score(test_data['Label'], test_data['Prediction']))
print('Recall:', recall_score(test_data['Label'], test_data['Prediction']))
print('Precision:', precision_score(test_data['Label'], test_data['Prediction']))
print('F1 Score:', f1_score(test_data['Label'], test_data['Prediction']))
print(confusion_matrix(test_data['Label'], test_data['Prediction']))
print(classification_report(test_data['Label'], test_data['Prediction']))
print('ROC AUC Score:', roc_auc_score(test_data['Label'], test_data['Prediction']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_data['Label'], test_data['Prediction'])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Plot precision-recall curve
precision, recall, thresholds = precision_recall_curve(test_data['Label'], test_data['Prediction'])
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Save the performance metrics to a text file
with open('performance_metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy_score(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'Recall: {recall_score(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'Precision: {precision_score(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'F1 Score: {f1_score(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'Confusion Matrix:\n{confusion_matrix(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'Classification Report:\n{classification_report(test_data["Label"], test_data["Prediction"])}\n')
    f.write(f'ROC AUC Score: {roc_auc_score(test_data["Label"], test_data["Prediction"])}\n')

# Save the ROC curve plot
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('roc_curve.png')

# Save the precision-recall curve plot
plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')

# Save the training and validation loss plot
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss.png')