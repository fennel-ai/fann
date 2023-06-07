try:
    import faiss
except Exception as ex:
    print("Please ensure faiss-cpu==1.7.3 is installed locally")
    raise ex
import pickle
import os
import numpy as np
import time
import argparse


def convert_raw_data_if_needed(data_dir: str, input_vec_path: str):
    input_file_name = os.path.split(input_vec_path)[1]
    input_file_no_ext = os.path.splitext(input_file_name)[0]
    processed_npz_path = os.path.join(data_dir, f"faiss_{input_file_no_ext}.npz")
    processed_pkl_path = os.path.join(data_dir, f"faiss_{input_file_no_ext}.pkl")
    if not os.path.exists(processed_npz_path) or not os.path.exists(processed_pkl_path):
        # We must actually process the data
        start = time.time()
        with open(input_vec_path, "r") as f:
            data = f.readlines()
        num_vectors, dim = [int(value) for value in data[0].split()]
        print(f"Running with {num_vectors} vectors and {dim} dimensions")
        sample_data = np.zeros((num_vectors, dim), dtype=np.float32)
        word_to_index_map = {}
        for i in range(len(sample_data)):
            entries = data[i + 1].split()
            word_to_index_map[entries[0]] = i
            for j in range(dim):
                sample_data[i,j] = float(entries[j+1])
        del data
        end = time.time()
        np.savez(processed_npz_path, x=sample_data)
        pickle.dump(word_to_index_map, open(processed_pkl_path, "wb"))
        print(f"Loaded in {sample_data.shape}-shape input data in {end-start} seconds")
    return processed_npz_path, processed_pkl_path


def load_data(npz_path: str, pkl_path: str):
    # Expects [convert_data] above to have been called once to pre-process the files
    sample_data = np.load(npz_path)["x"]
    word_to_index_map = pickle.load(open(pkl_path, "rb"))
    num_vectors, dim = sample_data.shape
    print(f"Running with {num_vectors} vectors and {dim} dimensions")
    index_to_word_map = {}
    for word, i in word_to_index_map.items():
        index_to_word_map[i] = word
    return sample_data, word_to_index_map, index_to_word_map


def index_and_check_runtime(
        data, ef_search = None, ef_construction = None, 
        max_node_size = 15, top_k = 20, save_all_distances_folder = None):
    print(f"ef_search={ef_search}, ef_construction={ef_construction}, max_node_size={max_node_size}, top_k={top_k}")
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
    search_data = data[np.random.choice(data.shape[0], size=1000)].reshape(1000, 1, -1)
    sch_start = time.time()
    for i in range(1000):
        # Take a random sample of vectors and just time search on FAISS
        index.search(search_data[i], top_k)
    sch_end = time.time()
    avg_sch_time = (sch_end - sch_start) / 1000
    print(f"Average time searching in bulk took {avg_sch_time} seconds")
    all_distances = []
    for vector in data:
        D, _ = index.search(vector.reshape(1,-1), top_k)
        all_distances.append(sum(D[0]**(1/2)) / len(D[0]))
    print(f"Average Euclidean Distance = {sum(all_distances)/len(all_distances)}")
    if save_all_distances_folder is not None:
        print("Saving all distances for future visualization")
        filename = f"faiss_{ef_search}_{ef_construction}_{max_node_size}_{top_k}.pkl"
        path = os.path.join(save_all_distances_folder, filename)
        pickle.dump(all_distances, open(path, "wb"))
    return index



def search_a_word_faiss(
        word, top_k, sample_data, word_to_index_map, index_to_word_map, index):
    search = np.zeros((1, sample_data.shape[1]), dtype=np.float32)
    search[0] = sample_data[word_to_index_map[word]]
    D, I = index.search(search, top_k)
    words = [index_to_word_map[I[0][i]] for i in range(I.shape[1])]
    return D[0]**(1/2), words


def search_a_word_exhaustive(
        word, top_k, sample_data, word_to_index_map, index_to_word_map):
    data = sample_data[word_to_index_map[word]]
    euc_distances = ((sample_data - data)**2).sum(axis=1)**(1/2)
    first_k_indices = euc_distances.argsort()[:top_k]
    distances = [euc_distances[idx] for idx in first_k_indices]
    words = [index_to_word_map[idx] for idx in first_k_indices]
    return distances, words


def display_comparison(
        word, top_k, sample_data, word_to_index_map, index_to_word_map, index):
    faiss_dist, faiss_words = search_a_word_faiss(
        word, top_k, sample_data, word_to_index_map, index_to_word_map, index)
    print(f"Word: {word}")
    print(f"FAISS Euclidean Dist: {faiss_dist}")
    print(f"FAISS Words: {faiss_words}")
    ex_dist, ex_words = search_a_word_exhaustive(
        word, top_k, sample_data, word_to_index_map, index_to_word_map)
    print(f"Exhaustive Euclidean Dist: {ex_dist}")
    print(f"Exhaustive Words: {ex_words}")
    print()


# Parse the location of the input file
parser = argparse.ArgumentParser(description="Run FAISS Benchmarking")
parser.add_argument("--data-dir", type=str, help="The data directory path")
parser.add_argument("--input-vec", type=str, help="The path to the input vec file")
args = parser.parse_args()

# Load in the sample data
# First time ever, run [convert_raw_data] and then you can save the conversion time
npz_path, pkl_path = convert_raw_data_if_needed(args.data_dir, args.input_vec)
sample_data, word_to_index_map, index_to_word_map = load_data(npz_path, pkl_path)

# On the default configurations, print all distances too
index = index_and_check_runtime(sample_data, save_all_distances_folder=args.data_dir)

# Now, evaluate FAISS results qualitatively
other_display_args = [20, sample_data, word_to_index_map, index_to_word_map, index]
display_comparison("river", *other_display_args)
display_comparison("war", *other_display_args)
display_comparison("love", *other_display_args)
display_comparison("education", *other_display_args)
