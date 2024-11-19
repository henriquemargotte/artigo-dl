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

import tensorflow as tf

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

data = pd.concat([data, data_ovs, data_meta])

# Set base seed
base_seed = 42

for run in range(1):
    print(f"Run {run}:")

    # Set the seed for reproducibility
    np.random.seed(base_seed + run)
    tf.random.set_seed(base_seed + run)
    random.seed(base_seed + run)

    print(data['Label'].unique())
    print(data['Label'].value_counts())

    #train_data, test_data = train_test_split(data, test_size=0.2, random_state=base_seed + run, stratify=data['Label'])
    # Define the desired number of samples for each label in training and test sets
    label_counts = {
        'Normal': (57956, 10468),
        'Probe': (12586, 2639),
        'DoS': (21622, 13778),
        'DDoS': (9440, 503),
        'BFA': (1007, 288),
        'BOTNET': (164, 0),
        'Web-Attack': (174, 18),
        'U2R': (14, 3)
    }

    # Split the data according to the specified counts
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for label, (train_count, test_count) in label_counts.items():
        label_data = data[data['Label'] == label]
        train_label_data = label_data.sample(n=train_count, random_state=base_seed + run)
        test_label_data = label_data.drop(train_label_data.index).sample(n=test_count, random_state=base_seed + run)
        train_data = pd.concat([train_data, train_label_data])
        test_data = pd.concat([test_data, test_label_data])
        print(f"{label}: Train {train_label_data.shape[0]}, Test {test_label_data.shape[0]}")
    print(f"Train: {train_data.shape[0]}, Test: {test_data.shape[0]}")
    # print amount of data per label
    print(train_data['Label'].value_counts())
    print(test_data['Label'].value_counts())

    train_data['Label'] = train_data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
    test_data['Label'] = test_data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

    # Reshape data for LSTM input
    train_data_values = np.expand_dims(train_data.drop(columns=['Label']).values, axis=1)
    test_data_values = np.expand_dims(test_data.drop(columns=['Label']).values, axis=1)

    print(train_data_values.shape)
    print(test_data_values.shape)
    print(train_data['Label'].value_counts())
    print(test_data['Label'].value_counts())
