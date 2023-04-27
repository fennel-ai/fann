use rand::prelude::IteratorRandom;
use std::io::BufRead;
use rayon::prelude::*;
mod search;

fn search_exhaustive<const N: usize>(all_data: &Vec<search::Vector<N>>, vector: &search::Vector<N>, top_k: i32) -> std::collections::HashSet<i32> {
    let enumerated_iter = all_data.iter().enumerate();
    let mut idx_sq_euc_dis: Vec<(usize, f32)> = enumerated_iter.map(|(i, can)| (i, can.sq_euc_dis(vector))).collect();
    idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    // Return a set of IDs corresponding to the closest matches
    let mut final_candidates = std::collections::HashSet::new();
    for i in 0..top_k as usize {
        final_candidates.insert(idx_sq_euc_dis[i].0 as i32);
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
        let embedding: [f32; N] = data_on_line_iter.map(
            |s| s.parse::<f32>().unwrap()).collect::<Vec<_>>().try_into().unwrap();
        all_data.push(search::Vector(embedding));
    }
}

fn search_approximate_as_hashset<const N: usize>(
    index: &search::ANNIndex<N>, vector: search::Vector<N>, top_k: i32
) -> std::collections::HashSet<i32> {
    let nearby_idx_and_distance = index.search_approximate(vector, top_k);
    let mut id_hashset = std::collections::HashSet::new();
    for &(idx, _) in nearby_idx_and_distance.iter() {
        id_hashset.insert(idx);
    }
    return id_hashset
}

fn build_benchmark_and_visualize_index<const N: usize>(
    my_input_data: &Vec<search::Vector<N>>,
    word_to_idx_mapping: &std::collections::HashMap<String, usize>,
    idx_to_word_mapping: &std::collections::HashMap<usize, String>,
    num_trees: i32, max_node_size: i32, top_k: i32,
    indices_of_interest: &Vec<i32>, words_to_visualize: &Vec<String>) -> Vec<std::collections::HashSet<i32>> {
    println!("dimensions={}, num_trees={}, max_node_size={}, top_k={}", N, num_trees, max_node_size, top_k);
    // Build the index
    let start = std::time::Instant::now();
    let my_ids: Vec<i32> = (0..my_input_data.len() as i32).collect();
    let index = search::ANNIndex::<N>::build_index(
        num_trees, max_node_size, &my_input_data, &my_ids);
    let duration = start.elapsed();
    println!("Build ANN index in {}-D in {:?}", N, duration);
    // Benchmark it with 1000 sequential queries
    let sample_idx: Vec<i32> = (0..my_input_data.len() as i32).choose_multiple(&mut rand::thread_rng(), 1000);
    let mut search_vectors: Vec<search::Vector<N>> = Vec::new();
    for idx in sample_idx {
        search_vectors.push(my_input_data[idx as usize]);
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
            println!("{}, distance={}", idx_to_word_mapping[&(idx as usize)], distance.sqrt());
        }
    }
    // For the indices of interest, find the top_k neighbours and return that data
    let start = std::time::Instant::now(); 
    let index_results = indices_of_interest.par_iter().map(
        | &idx | search_approximate_as_hashset(&index, my_input_data[idx as usize], top_k)).collect();
    let duration = start.elapsed();
    println!("Collected sample of index quality in {:?}", duration);
    return index_results;
}

fn calculate_metrics_on_index_result<const N: usize>(
    all_embedding_data: &Vec<search::Vector<N>>, 
    exhaustive_results: &Vec<std::collections::HashSet<i32>>,
    index_results: &Vec<std::collections::HashSet<i32>>
) {
    // We wish to calculate the average Euclidean distance and the recall@k with k=20. However, running
    // this for all 1 mil embeddings will make the code run long. We thus take a sample size of 30k out of
    // the total 1 mil to get an estimate for these values. The distances are computed against the full 
    // 300-D embeddings and not the reduced dimensionality in the index.
    let start = std::time::Instant::now();
    let mut total_euc_dist = 0.0;
    let mut total_recall_pct = 0.0;
    for (i, neighbours) in index_results.iter().enumerate() {
        // Ignore the distance here since we must compute it against the full 300-D embedding
        let mut sum_of_dist_to_neighbours = 0.0;
        let mut num_matches_with_brute_results = 0.0;
        for &neighbour_id in neighbours.iter() {
            sum_of_dist_to_neighbours += all_embedding_data[i].sq_euc_dis(&all_embedding_data[neighbour_id as usize]).sqrt();
            if exhaustive_results[i].contains(&neighbour_id) {
                num_matches_with_brute_results += 1.0;
            }
        }
        total_euc_dist += sum_of_dist_to_neighbours / neighbours.len() as f32;
        total_recall_pct += num_matches_with_brute_results / neighbours.len() as f32;
    }
    let average_dist =  total_euc_dist / exhaustive_results.len() as f32;
    let average_recall = total_recall_pct / exhaustive_results.len() as f32;
    let duration = start.elapsed();
    println!("Average Euclidean Distance = {}, Average Recall% = {} in {:?}", average_dist, average_recall, duration);
}

fn main() {
    const DIM: usize = 300;
    const TOP_K: i32 = 20;
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
    search_exhaustive::<DIM>(&my_input_data, &my_input_data[0], TOP_K);
    let duration = start.elapsed();
    println!("Found vectors via brute-search in {}-D in {:?}", DIM, duration);
    // Take 1000 random vectors from the input data and find its TOP_K nearest neighbors using the
    // exhaustive/brute-force approach. This allows us to calculate recall for our implementations.
    // This is a list of randomly chosen indices from our main embedding set - we use a subset to
    // estimate our metrics due to the computation cost.
    let start = std::time::Instant::now();
    rayon::ThreadPoolBuilder::new().num_threads(5).build_global().unwrap();
    let sample_idx: Vec<i32> = (0..my_input_data.len() as i32).choose_multiple(&mut rand::thread_rng(), 1000);
    // Make a vector of hashsets where hashset at position i represents the exhaustive nearest neighbours
    // for the embedding at position idx in my_input_data, where idx is at position i in sample_idx.
    let exhaustive_results: Vec<std::collections::HashSet<i32>> = sample_idx.par_iter().map(
        |&idx| search_exhaustive::<DIM>(&my_input_data, &my_input_data[idx as usize], TOP_K)).collect();
    let duration = start.elapsed();
    println!("Found exhaustive neighbors for sample to calculate recall in {:?}", duration);
    // Main parameters
    let input_words = ["river", "war", "love", "education"];
    let words_to_visualize: Vec<String> = input_words.into_iter().map(|x| x.to_owned()).collect();
    let index_results = build_benchmark_and_visualize_index::<DIM>(
        &my_input_data, &word_to_idx_mapping, &idx_to_word_mapping,
        3, 15, TOP_K, &sample_idx, &words_to_visualize);
    calculate_metrics_on_index_result::<DIM>(&my_input_data, &exhaustive_results, &index_results);
    // Try some other parameters. New values for max_node_size, num_trees at dim=300. See how we can make it
    // better in its accuracy/ quality.
    let no_words: Vec<String> = Vec::new();
    for num_trees in [3, 9, 15] {
        for max_node_size in [5, 15, 30] {
            let index_results = build_benchmark_and_visualize_index::<DIM>(
                &my_input_data, &word_to_idx_mapping, &idx_to_word_mapping,
                num_trees, max_node_size, TOP_K, &sample_idx, &no_words);
            calculate_metrics_on_index_result::<DIM>(&my_input_data, &exhaustive_results, &index_results);
        }
    }
}
