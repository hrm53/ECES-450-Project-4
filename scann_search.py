import numpy as np
import h5py
import os
import requests
import tempfile
import time
import scann
import pandas as pd
import random

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

test_embedding = np.load("test_embedding.npy")
db_embedding = np.load("db_embedding.npy")

X_train = pd.read_csv("metadata_db.csv")
X_test = pd.read_csv("metadata_test.csv")

print("Partition on")
print("AH scoring on, quantize off")
print("reorder off")

print("test embedding shape: ", test_embedding.shape)
print("database embedding shape: ", db_embedding.shape)

#random seed test
random.seed(10)


#normalize dataset -> create searcher
start = time.time()
normalized_dataset = db_embedding / np.linalg.norm(db_embedding, axis=1)[:, np.newaxis]
record_count = len(X_train)
searcher = scann.scann_ops_pybind.builder(normalized_dataset, 100, "dot_product").score_ah(
    2, anisotropic_quantization_threshold=0.2).build()
        
#searcher = scann.scann_ops_pybind.builder(normalized_dataset, 100, "dot_product").tree(
#        num_leaves=2000, num_leaves_to_search=10, training_sample_size=250000).score_ah(
#        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
end = time.time()
print("index latency (s):", (end - start))  

#redo for top hits
start = time.time()
neighbors, distances = searcher.search_batched(test_embedding, final_num_neighbors=10)
end = time.time()
print("query latency (s):", (end - start))
#get predictions
predicted_label = X_train.iloc[neighbors[:,0].reshape(-1)]["gene"]

print('predicted length: ', len(predicted_label))
print('number of predicted length: ', len(neighbors[0]))

# Calculate accuracy
accuracy = accuracy_score(X_test["gene"], predicted_label)
print(f'Accuracy: {accuracy}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(X_test["gene"], predicted_label)

# Calculate MCC, sensitivity, and specificity
mcc = matthews_corrcoef(X_test["gene"], predicted_label)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')

#np.save("scann_neighbors_final", neighbors)  
#np.save("scann_distances_final", distances)  
