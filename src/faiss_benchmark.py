# Run python3 -m pip install faiss-cpu. faiss-cpu in Python is a thin-wrapper around the C++ API.
# Download the wikinews data (https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
# and place the unzipped file in the data folder.
import faiss
import pickle
import os
import numpy as np
import time

MAX_NODE_SIZE = 15
TOP_K = 20
NUM_VECTORS, DIM = None, None
DATA_DIR = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "data")


def convert_raw_data():
    start = time.time()
    with open(f"{DATA_DIR}/wiki-news-300d-1M.vec", "r") as f:
        data = f.readlines()
    global NUM_VECTORS, DIM
    NUM_VECTORS, DIM = [int(value) for value in data[0].split()]
    print(f"Running with {NUM_VECTORS} vectors and {DIM} dimensions")
    sample_data = np.zeros((NUM_VECTORS, DIM), dtype=np.float32)
    word_to_index_map = {}
    for i in range(len(sample_data)):
        entries = data[i + 1].split()
        word_to_index_map[entries[0]] = i
        for j in range(DIM):
            sample_data[i,j] = float(entries[j+1])
    del data
    end = time.time()
    np.savez(f"{DATA_DIR}/wiki-news-array.npz", x=sample_data)
    pickle.dump(word_to_index_map, open(f"{DATA_DIR}/wiki-news-map.pkl", "wb"))
    print(f"Loaded in {sample_data.shape}-shape input data in {end-start} seconds")


def load_data():
    # Expects [convert_data] above to have been called once to pre-process the files
    sample_data = np.load(f"{DATA_DIR}/wiki-news-array.npz")["x"]
    word_to_index_map = pickle.load(open(f"{DATA_DIR}/wiki-news-map.pkl", "rb"))
    global NUM_VECTORS, DIM
    NUM_VECTORS, DIM = sample_data.shape
    print(f"Running with {NUM_VECTORS} vectors and {DIM} dimensions")
    index_to_word_map = {}
    for word, i in word_to_index_map.items():
        index_to_word_map[i] = word
    return sample_data, word_to_index_map, index_to_word_map

# Load in the sample data
# First time ever, run [convert_raw_data] and then you can save the conversion time
# convert_raw_data()
sample_data, word_to_index_map, index_to_word_map = load_data()

def index_and_check_runtime(data):
    # Index this data into Faiss
    idx_start = time.time()
    index = faiss.IndexHNSWFlat(data.shape[1], MAX_NODE_SIZE)
    index.add(data)
    idx_end = time.time()
    print(f"Indexed data into HNSWFlat Index in {idx_end-idx_start} seconds")
    total_sch_time = 0
    for _ in range(10):
        # Take a random sample of vectors and just time search on FAISS
        search_vectors_idx = np.random.choice(data.shape[0], 100, replace=False)
        search_vectors = data[search_vectors_idx]
        sch_start = time.time()
        index.search(search_vectors, TOP_K)
        sch_end = time.time()
        total_sch_time += sch_end - sch_start
    avg_sch_time = total_sch_time / (100 * 10)
    print(f"Average time searching in bulk took {avg_sch_time} seconds")
    return index, idx_end - idx_start, avg_sch_time


index, _, _ = index_and_check_runtime(sample_data)


def search_a_word_faiss(word):
    search = np.zeros((1, DIM), dtype=np.float32)
    search[0] = sample_data[word_to_index_map[word]]
    D, I = index.search(search, TOP_K)
    words = [index_to_word_map[I[0][i]] for i in range(I.shape[1])]
    return D[0], words


def search_a_word_exhaustive(word):
    data = sample_data[word_to_index_map[word]]
    sq_euc_distances = ((sample_data - data)**2).sum(axis=1)
    first_k_indices = sq_euc_distances.argsort()[:TOP_K]
    distances = [sq_euc_distances[idx] for idx in first_k_indices]
    words = [index_to_word_map[idx] for idx in first_k_indices]
    return distances, words


def display_comparison(word):
    faiss_dist, faiss_words = search_a_word_faiss(word)
    print(f"Word: {word}")
    print(f"FAISS Euclidean Dist: {faiss_dist}")
    print(f"FAISS Words: {faiss_words}")
    ex_dist, ex_words = search_a_word_exhaustive(word)
    print(f"Exhaustive Euclidean Dist: {ex_dist}")
    print(f"Exhaustive Words: {ex_words}")
    print()


# Now, evaluate FAISS results qualitatively
display_comparison("river")
display_comparison("war")
display_comparison("love")
display_comparison("education")


# Now, we investigate the scalability of search in term of the dimensions and num vectors
# First, vary the dimensions from 60 to 300 in increments of 20 to see how the time to
# index and search changes. We trivially take the first X dimensions of FastText embeddings.
# Next, we vary the number of vectors at dim 300 from 200k to 1m in increments of 100k
# and track changes in the time.
num_dims_to_check = np.arange(60, 301, 20)
indexing_time_by_num_dims = [None for _ in num_dims_to_check]
search_time_by_num_dims = [None for _ in num_dims_to_check]
for i, dim in enumerate(num_dims_to_check):
    print(f"Investigating indexing and search time on dims={dim}")
    _, idx_time, sch_time = index_and_check_runtime(sample_data[:,:dim])
    indexing_time_by_num_dims[i] = idx_time
    search_time_by_num_dims[i] = sch_time
print(f"Indexing Time: {indexing_time_by_num_dims}, Search Time: {search_time_by_num_dims}")

num_vectors_to_check = np.arange(200000, 1000001, 100000)
indexing_time_by_num_vectors = [None for _ in num_vectors_to_check]
search_time_by_num_vectors = [None for _ in num_vectors_to_check]
for i, vec in enumerate(num_vectors_to_check):
    print(f"Investigating indexing and search time on num_vectors={vec}")
    _, idx_time, sch_time = index_and_check_runtime(sample_data[:vec,:])
    indexing_time_by_num_vectors[i] = idx_time
    search_time_by_num_vectors[i] = sch_time
print(f"Indexing Time: {indexing_time_by_num_vectors}, Search Time: {search_time_by_num_vectors}")
