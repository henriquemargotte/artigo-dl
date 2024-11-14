import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import kagglehub

# Load dataset
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")
file_path = path + '/InSDN_DatasetCSV/Normal_data.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Flow ID'])
data = data.set_index('Timestamp')
scaler = MinMaxScaler(feature_range=(-1, 1))
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

# Split data into features and labels
X = data.drop(columns=['Label']).values
y = data['Label'].values

# Train One-Class SVM on normal data
oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.4)
oc_svm.fit(X[y == 0])  # Train only on "Normal" data

# Predict anomalies
svm_pred = oc_svm.predict(X)
svm_pred = np.where(svm_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

# Calculate performance metrics
accuracy = accuracy_score(y, svm_pred)
recall = recall_score(y, svm_pred)
precision = precision_score(y, svm_pred)
f1 = f1_score(y, svm_pred)
roc_auc = roc_auc_score(y, svm_pred)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y, svm_pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for One-Class SVM Anomaly Detection')
plt.legend(loc="lower right")
plt.show()
