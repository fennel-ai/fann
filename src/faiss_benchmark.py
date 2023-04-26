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

def index_and_check_runtime(
        data, ef_search = None, ef_construction = None, max_node_size = MAX_NODE_SIZE):
    # Index this data into Faiss
    idx_start = time.time()
    index = faiss.IndexHNSWFlat(data.shape[1], max_node_size)
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    if ef_construction is not None:
        index.hnsw.efConstruction = ef_construction
    index.add(data)
    idx_end = time.time()
    idx_time = idx_end - idx_start
    print(f"Indexed data into HNSWFlat Index in {idx_time} seconds")
    times = []
    for _ in range(1000):
        # Take a random sample of vectors and just time search on FAISS
        search_vector = data[np.random.choice(data.shape[0])].reshape(1,-1)
        sch_start = time.time()
        index.search(search_vector, TOP_K)
        sch_end = time.time()
        times.append(sch_end - sch_start)
    times.sort()
    avg_sch_time = sum(times)/len(times)
    p95_sch_time = times[950]
    print(f"Average time searching in bulk took {avg_sch_time} seconds")
    return index, idx_time, avg_sch_time, p95_sch_time


index, _, _, _ = index_and_check_runtime(sample_data)


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
    print(f"FAISS Squared Euclidean Dist: {faiss_dist}")
    print(f"FAISS Words: {faiss_words}")
    ex_dist, ex_words = search_a_word_exhaustive(word)
    print(f"Exhaustive Squared Euclidean Dist: {ex_dist}")
    print(f"Exhaustive Words: {ex_words}")
    print()


# Now, evaluate FAISS results qualitatively
display_comparison("river")
display_comparison("war")
display_comparison("love")
display_comparison("education")


# Now, we investigate the effects of efSearch, efConstruction, num_dimensions, and num_vectors
# on indexing / search time. We vary efSearch from 16 to 1024 in some powers of 2, efConstruction from
# 32 to 256 in powers of 2, num_dimensions from 60 to 300 in increments of 20 (trivially taking
# the first X dimensions of FastText embeddings).
num_dims_to_check = np.arange(60, 301, 40)
ef_search_to_check = [16, 64, 128, 512]
ef_construction_to_check = [32, 64, 128]
final_data = {}
for dim in num_dims_to_check:
    for ef_search in ef_search_to_check:
        for ef_construction in ef_construction_to_check:
            _, idx_time, avg_sch_time, p95_sch_time = index_and_check_runtime(
                sample_data[:,:dim], ef_search, ef_construction)
            final_data[(dim, ef_search, ef_construction)] = (idx_time, avg_sch_time, p95_sch_time)
pickle.dump(final_data, open(f"{DATA_DIR}/faiss-benchmarks.pkl", "wb"))
