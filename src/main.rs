use rand::prelude::IteratorRandom;
use std::io::BufRead;
mod search;

fn search_exhaustive<const N: usize>(all_data: &Vec<search::Vector<N>>, vector: &search::Vector<N>, top_k: i32) -> Vec<search::Vector<N>> {
    let enumerated_iter = all_data.iter().enumerate();
    let mut idx_sq_euc_dis: Vec<(usize, f32)> = enumerated_iter.map(|(i, can)| (i, can.sq_euc_dis(vector))).collect();
    idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut final_candidates: Vec<search::Vector<N>> = Vec::new();
    for i in 0..top_k as usize {
        final_candidates.push(all_data[idx_sq_euc_dis[i].0])
    }
    return final_candidates
}

fn load_raw_wiki_data<const N: usize>(
    filename: &str, all_data: &mut Vec<search::Vector<N>>,
    word_to_idx_mapping: &mut std::collections::HashMap<String, usize>,
    idx_to_word_mapping: &mut std::collections::HashMap<usize, String>) {
    // wiki-news has 999,994 vectors in 300 dimensions
    let file = std::fs::File::open(filename).expect("Should have been able to read the file");
    let reader = std::io::BufReader::new(file);
    let mut cur_idx: usize = 0;
    // We skip the first line that simply has metadata
    for maybe_line in reader.lines().skip(1) {
        let line = maybe_line.expect("Should decode the line");
        let mut data_on_line_iter = line.split_whitespace();
        let word = data_on_line_iter.next().expect("Each line begins with a word");
        // Update the mappings
        word_to_idx_mapping.insert(word.to_owned(), cur_idx);
        idx_to_word_mapping.insert(cur_idx, word.to_owned());
        cur_idx += 1;
        // Parse the vector. Everything except the word on the line is the vector
        let embedding: [f32; N] = data_on_line_iter.map(|s| s.parse::<f32>().unwrap()).collect::<Vec<_>>().try_into().unwrap();
        all_data.push(search::Vector(embedding));
    }
}

fn build_benchmark_and_visualize_index<const N: usize>(
    my_input_data: &Vec<search::Vector<N>>,
    word_to_idx_mapping: &std::collections::HashMap<String, usize>,
    idx_to_word_mapping: &std::collections::HashMap<usize, String>,
    num_trees: i32, max_node_size: i32, top_k: i32, words_to_visualize: &Vec<String>) {
    // Build the index
    let start = std::time::Instant::now();
    let my_ids: Vec<i32> = (0..my_input_data.len() as i32).collect();
    let index = search::ANNIndex::<N>::build_an_index(
        num_trees, max_node_size, &my_input_data, &my_ids);
    let duration = start.elapsed();
    println!("Build ANN index in {}-D in {:?}", N, duration);
    // Benchmark it with 1000 sequential queries
    let sample_idx: Vec<usize> = (0..my_input_data.len()).choose_multiple(&mut rand::thread_rng(), 1000);
    let mut search_vectors: Vec<search::Vector<N>> = Vec::new();
    for idx in sample_idx {
        search_vectors.push(my_input_data[idx]);
    }
    let start = std::time::Instant::now();
    for i in 0..1000 {
        index.search_approximate(search_vectors[i], top_k);
    }
    let duration = start.elapsed() / 1000;
    println!("Bulk ANN-search in {}-D has average time {:?}", N, duration);
    // Visualize the results for some words
    for word in words_to_visualize.iter() {
        println!("Currently visualizing {}", word);
        let word_index = word_to_idx_mapping[word];
        let embedding = my_input_data[word_index];
        let nearby_idx_and_distance = index.search_approximate(embedding, top_k);
        for &(idx, distance) in nearby_idx_and_distance.iter() {
            println!("{}, sq distance={}", idx_to_word_mapping[&(idx as usize)], distance);
        }
    }
}

fn main() {
    const DIM: usize = 300;
    const NUM_TREES: i32 = 3;
    const TOP_K: i32 = 20;
    const MAX_NODE_SIZE : i32 = 15;
    // Parse the data from wiki-news
    let start = std::time::Instant::now();
    let mut my_input_data: Vec<search::Vector<DIM>> = Vec::new();
    let mut word_to_idx_mapping: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut idx_to_word_mapping: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    load_raw_wiki_data::<DIM>(
        "data/wiki-news-300d-1M.vec", &mut my_input_data, 
        &mut word_to_idx_mapping, &mut idx_to_word_mapping);
    let duration = start.elapsed();
    println!("Parsed {} vectors in {}-D in {:?}", my_input_data.len(), DIM, duration);
    // Try the naive exact-search for TOP_K elements
    let start = std::time::Instant::now();
    let brute_results = search_exhaustive::<DIM>(&my_input_data, &my_input_data[0], TOP_K);
    let duration = start.elapsed();
    println!("Found {} vectors via brute-search in {}-D in {:?}", brute_results.len(), DIM, duration);
    // Main parameters
    let input_words = ["river", "war", "love", "education"];
    let words_to_visualize: Vec<String> = input_words.into_iter().map(|x| x.to_owned()).collect();
    build_benchmark_and_visualize_index::<DIM>(
        &my_input_data, &word_to_idx_mapping, &idx_to_word_mapping,
        NUM_TREES, MAX_NODE_SIZE, TOP_K, &words_to_visualize);
    // See how run-times change based on parameters
    const DIM_60: usize = 60;
    let mut data_dim_60: Vec<search::Vector<DIM_60>> = Vec::new();
    const DIM_120: usize = 120;
    let mut data_dim_120: Vec<search::Vector<DIM_120>> = Vec::new();
    const DIM_200: usize = 200;
    let mut data_dim_200: Vec<search::Vector<DIM_200>> = Vec::new();
    for vector in my_input_data.iter() {
        let dim_60: [f32; DIM_60] = (vector.0)[0..DIM_60].try_into().unwrap();
        let dim_120: [f32; DIM_120] = (vector.0)[0..DIM_120].try_into().unwrap();
        let dim_200: [f32; DIM_200] = (vector.0)[0..DIM_200].try_into().unwrap();
        data_dim_60.push(search::Vector(dim_60));
        data_dim_120.push(search::Vector(dim_120));
        data_dim_200.push(search::Vector(dim_200));
    }
    let no_words: Vec<String> = Vec::new();
    for num_trees in [3, 5, 9, 15] {
        for max_node_size in [5, 15, 30] {
            build_benchmark_and_visualize_index::<DIM_60>(
                &data_dim_60, &word_to_idx_mapping, &idx_to_word_mapping,
                num_trees, max_node_size, TOP_K, &no_words);
        }
    }
    for num_trees in [3, 5, 9, 15] {
        for max_node_size in [5, 15, 30] {
            build_benchmark_and_visualize_index::<DIM_120>(
                &data_dim_120, &word_to_idx_mapping, &idx_to_word_mapping,
                num_trees, max_node_size, TOP_K, &no_words);
        }
    }
    for num_trees in [3, 5, 9, 15] {
        for max_node_size in [5, 15, 30] {
            build_benchmark_and_visualize_index::<DIM_200>(
                &data_dim_200, &word_to_idx_mapping, &idx_to_word_mapping,
                num_trees, max_node_size, TOP_K, &no_words);
        }
    }
    for num_trees in [3, 5, 9, 15] {
        for max_node_size in [5, 15, 30] {
            build_benchmark_and_visualize_index::<DIM>(
                &my_input_data, &word_to_idx_mapping, &idx_to_word_mapping,
                num_trees, max_node_size, TOP_K, &no_words);
        }
    }
}
