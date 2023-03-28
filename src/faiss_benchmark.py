# Run python3 -m pip install faiss-cpu. faiss-cpu in Python is a thin-wrapper around the C++ API.
import faiss
import numpy as np
import time
from math import floor, ceil

DIM = 30
MAX_NODE_SIZE = 15
NUM_VECTORS = 1000000
TOP_K = 20

start = time.time()
index = faiss.IndexHNSWFlat(DIM, MAX_NODE_SIZE)
sample_data = np.random.normal(size=(floor(NUM_VECTORS/2), DIM)).astype(np.float32)
index.add(sample_data)
sample_data = np.random.normal(size=(ceil(NUM_VECTORS/2), DIM)).astype(np.float32)
index.add(sample_data)
end = time.time()
print(f"Creating and indexing data took {end-start} seconds")

sample_vectors = np.random.normal(size=(20, DIM)).astype(np.float32)
start = time.time()
D, I = index.search(sample_vectors, TOP_K)
end = time.time()
print(f"Average time searching in bulk took {(end-start) / 20} seconds")
