import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt

distance_order = np.load("scann_distance_order2.npy")

# Initialize UMAP
reducer = umap.UMAP(random_state=42)

# Fit UMAP on the entire dataset

reduced_data_umap = reducer.fit_transform(distance_order)


np.save('umap_results_scann_distance_order2.npy', reduced_data_umap)