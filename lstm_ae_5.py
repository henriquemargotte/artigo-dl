import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import os
import kagglehub

# Setup directories for saving results
os.makedirs('plots', exist_ok=True)
os.makedirs('performance', exist_ok=True)

# Placeholder for storing metrics
results = {
    'accuracy': [],
    'recall': [],
    'precision': [],
    'f1_score': [],
    'roc_auc': []
}

# Load and process data only once
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")
file_path = path + '/InSDN_DatasetCSV/Normal_data.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data = data.set_index('Timestamp')
scaler = StandardScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

file_path = path + '/InSDN_DatasetCSV/OVS.csv'
data_ovs = pd.read_csv(file_path)
data_ovs = data_ovs.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data_ovs = data_ovs.set_index('Timestamp')
data_ovs[data_ovs.columns[:-1]] = scaler.fit_transform(data_ovs[data_ovs.columns[:-1]])
data_ovs = data_ovs.sample(frac=0.1, random_state=42)
data_ovs['Label'] = data_ovs['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data = pd.concat([test_data, data_ovs])
train_data_values = np.expand_dims(train_data.drop(columns=['Label']).values, axis=1)
test_data_values = np.expand_dims(test_data.drop(columns=['Label']).values, axis=1)

# Define model architecture (same across runs)
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
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Train and evaluate five times
for i in range(5):
    print(f"Run {i+1}:")

    # Train the model
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    history = model.fit(train_data_values, train_data_values, epochs=100, batch_size=32, callbacks=callbacks, validation_split=0.1, verbose=0)

    # Plot and save training history
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss (Run {i+1})')
    plt.savefig(f'plots/training_validation_loss_run_{i+1}.png')
    plt.close()

    # Anomaly threshold calculation
    train_pred = model.predict(train_data_values)
    train_loss = np.mean(np.abs(train_pred - train_data_values), axis=(1, 2))
    threshold = np.mean(train_loss) + 3 * np.std(train_loss)

    # Evaluate on test data
    test_pred = model.predict(test_data_values)
    test_loss = np.mean(np.abs(test_pred - test_data_values), axis=(1, 2))
    test_data['Loss'] = test_loss
    test_data['Prediction'] = test_data['Loss'] > threshold

    # Calculate metrics and append to results
    accuracy = accuracy_score(test_data['Label'], test_data['Prediction'])
    recall = recall_score(test_data['Label'], test_data['Prediction'])
    precision = precision_score(test_data['Label'], test_data['Prediction'])
    f1 = f1_score(test_data['Label'], test_data['Prediction'])
    roc_auc = roc_auc_score(test_data['Label'], test_data['Prediction'])

    results['accuracy'].append(accuracy)
    results['recall'].append(recall)
    results['precision'].append(precision)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc_auc)

    print(f"Run {i+1} Results - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}, ROC AUC: {roc_auc}\n")

# Calculate and plot the mean of each metric across the five runs
mean_accuracy = np.mean(results['accuracy'])
mean_recall = np.mean(results['recall'])
mean_precision = np.mean(results['precision'])
mean_f1_score = np.mean(results['f1_score'])
mean_roc_auc = np.mean(results['roc_auc'])

print("Final averaged metrics over 5 runs:")
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean Precision: {mean_precision}")
print(f"Mean F1 Score: {mean_f1_score}")
print(f"Mean ROC AUC: {mean_roc_auc}")

# Plot average metrics
plt.figure()
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC']
mean_values = [mean_accuracy, mean_recall, mean_precision, mean_f1_score, mean_roc_auc]
plt.bar(metrics, mean_values)
plt.title('Average Performance Metrics over 5 Runs')
plt.savefig('performance/average_performance_metrics.png')
plt.close()

# Save each run's performance metrics to a text file
with open('performance/performance_metrics.txt', 'w') as f:
    for i in range(5):
        f.write(f"Run {i+1}:\n")
        f.write(f"  Accuracy: {results['accuracy'][i]}\n")
        f.write(f"  Recall: {results['recall'][i]}\n")
        f.write(f"  Precision: {results['precision'][i]}\n")
        f.write(f"  F1 Score: {results['f1_score'][i]}\n")
        f.write(f"  ROC AUC: {results['roc_auc'][i]}\n\n")
    f.write("Final Averaged Metrics:\n")
    f.write(f"  Mean Accuracy: {mean_accuracy}\n")
    f.write(f"  Mean Recall: {mean_recall}\n")
    f.write(f"  Mean Precision: {mean_precision}\n")
    f.write(f"  Mean F1 Score: {mean_f1_score}\n")
    f.write(f"  Mean ROC AUC: {mean_roc_auc}\n")
