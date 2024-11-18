import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, auc, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import kagglehub

# import tsne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.metrics import confusion_matrix
import tensorflow as tf

def plot_and_compare(dataX, dataY, predy, title, save_as_pdf=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot confusion matrix
    sns.heatmap(confusion_matrix(dataY, predy), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{title}\nConfusion Matrix')

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, verbose=1, n_jobs=-1)
    X_tsne = tsne.fit_transform(dataX)

    # t-SNE plot colored by misclassification
    misclassified = predy != dataY
    missclassified_labels = []
    for i in range(len(predy)):
        if misclassified[i]:
            missclassified_labels.append('misclassified {}->{}'.format(dataY[i], predy[i]))
        else:
            missclassified_labels.append('correct')

    for label in np.unique(missclassified_labels):
        idxs = np.where(np.array(missclassified_labels) == label)
        axes[1].scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], label=label, alpha=0.6)
    axes[1].set_title('t-SNE: Points colored by misclassification')
    axes[1].legend()

    # t-SNE plot colored by true class labels
    scatter = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataY, cmap='tab10', alpha=0.6)
    axes[2].set_title('t-SNE: Points colored by true labels')
    plt.colorbar(scatter, ax=axes[2])

    if save_as_pdf:
        plt.savefig(f'{title}.png')
    else:
        plt.tight_layout()
        plt.show()

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

from sklearn.preprocessing import MinMaxScaler

# Load and process data only once
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")
file_path = path + '/InSDN_DatasetCSV/Normal_data.csv'
data = pd.read_csv(file_path)
file_path = path + '/InSDN_DatasetCSV/OVS.csv'
data_ovs = pd.read_csv(file_path)
file_path = path + '/InSDN_DatasetCSV/metasploitable-2.csv'
data_meta = pd.read_csv(file_path)
data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID', 'Timestamp'])
scaler = MinMaxScaler(feature_range=(-1, 1))
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

data_ovs = data_ovs.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID', 'Timestamp'])
data_ovs[data_ovs.columns[:-1]] = scaler.transform(data_ovs[data_ovs.columns[:-1]])

data_meta = data_meta.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID', 'Timestamp'])
data_meta[data_meta.columns[:-1]] = scaler.transform(data_meta[data_meta.columns[:-1]])

#data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
#data_ovs['Label'] = data_ovs['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
#data_meta['Label'] = data_meta['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

data = pd.concat([data, data_ovs, data_meta])

# Set base seed
base_seed = 42

for run in range(5):
    print(f"Run {run}:")

    # Set the seed for reproducibility
    np.random.seed(base_seed + run)
    tf.random.set_seed(base_seed + run)
    random.seed(base_seed + run)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=base_seed + run, stratify=data['Label'])
    train_data['Label'] = train_data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
    test_data['Label'] = test_data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

    # Reshape data for LSTM input
    train_data_values = np.expand_dims(train_data.drop(columns=['Label']).values, axis=1)
    test_data_values = np.expand_dims(test_data.drop(columns=['Label']).values, axis=1)

    # Define encoder-decoder LSTM Autoencoder model for this run
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

    output = Dense(train_data_values.shape[2], activation='linear')(decoded)

    # Encoder model to extract compressed features
    encoder = Model(inputs=inputs, outputs=encoded)

    # Complete autoencoder model
    autoencoder = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Train the autoencoder
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    history = autoencoder.fit(
        train_data_values,
        train_data_values,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        validation_split=0.1, verbose=0)

    # Plot and save training history
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss AE (Run {run})')
    plt.savefig(f'plots/training_validation_loss_AE_run_{run}.png')
    plt.close()

    # Extract compressed features using the encoder
    train_compressed = encoder.predict(train_data_values)
    test_compressed = encoder.predict(test_data_values)

    # Train the One-Class SVM on normal data compressed features
    oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.4)
    oc_svm.fit(train_compressed[train_data['Label'] == 0])

    # Detect anomalies in test data using the SVM model
    svm_pred = oc_svm.predict(test_compressed)
    svm_pred = np.where(svm_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

    plot_and_compare(test_compressed, test_data['Label'].to_numpy(), svm_pred, f'One-Class SVM Run {run}', save_as_pdf=True)

    # Calculate performance metrics and append to results
    accuracy = accuracy_score(test_data['Label'], svm_pred)
    recall = recall_score(test_data['Label'], svm_pred)
    precision = precision_score(test_data['Label'], svm_pred)
    f1 = f1_score(test_data['Label'], svm_pred)
    roc_auc = roc_auc_score(test_data['Label'], svm_pred)

    results['accuracy'].append(accuracy)
    results['recall'].append(recall)
    results['precision'].append(precision)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc_auc)

    print(f"Run {run} Results - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}, ROC AUC: {roc_auc}\n")

    # Plot the ROC curve
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Run {run})')
    fpr, tpr, _ = roc_curve(test_data['Label'], svm_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.savefig(f'plots/roc_curve_run_{run}.png')
    plt.close()

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
    for i in range(len(results['accuracy'])):
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