use rand::{prelude::SliceRandom, Rng};

#[derive(Eq, PartialEq, Hash)]
struct HashKey<const N: usize>([u32; N]);

#[derive(Copy, Clone)]
struct Vector<const N: usize> { coordinates: [f32; N] }
impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped_iter = self.coordinates.iter().zip(vector.coordinates).map(|(a, b)| b - a);
        let coordinates: [f32; N] = mapped_iter.collect::<Vec<_>>().try_into().unwrap();
        return Vector{ coordinates: coordinates };
    }
    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped_iter = self.coordinates.iter().zip(vector.coordinates).map(|(a, b)| (a + b) / 2.0);
        let coordinates: [f32; N] = mapped_iter.collect::<Vec<_>>().try_into().unwrap();
        return Vector{ coordinates: coordinates };
    }
    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.coordinates.iter().zip(vector.coordinates);
        return zipped_iter.map(|(a, b)| a * b).sum::<f32>();
    }
    pub fn to_hashkey(&self) -> HashKey<N> {
        // f32 in Rust doesn't implement hash - we use byte representation to deduplicate which is unsafe in that it
        // cannot differentiate ~16 mil representations of NaN under IEEE-754 but safe for our demonstration
        let bit_iter = self.coordinates.iter().map(|a| a.to_bits());
        let u32_data: [u32; N] = bit_iter.collect::<Vec<_>>().try_into().unwrap();
        return HashKey::<N>(u32_data);
    }
    pub fn sq_euc_dis(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.coordinates.iter().zip(vector.coordinates);
        return zipped_iter.map(|(a, b)| (a - b).powi(2)).sum();
    }
}

struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}
impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }
}

fn build_hyperplane_bwn_two_random_points<const N: usize>(all_points: &Vec<Vector<N>>) -> (HyperPlane<N>, Vec<Vector<N>>, Vec<Vector<N>>) {
    let sample: Vec<_> = all_points.choose_multiple(&mut rand::thread_rng(), 2).collect();
    // We use implicit Cartesian equation for hyperplane n * (x - x_0) = 0. n (normal vector) is the coefs x_1 to x_n
    let coefficients = sample[0].subtract_from(sample[1]);
    let point_on_plane = sample[0].avg(sample[1]);
    let constant = - coefficients.dot_product(&point_on_plane);
    // Figure out which points lie above and below
    let mut above: Vec<Vector<N>> = Vec::new();
    let mut below: Vec<Vector<N>> = Vec::new();
    let hyperplane = HyperPlane::<N> { coefficients: coefficients, constant: constant };
    for &point in all_points.iter() {
        if hyperplane.point_is_above(&point) { above.push(point) } else { below.push(point) };
    }
    return (hyperplane, above, below);
}

struct ANNIndex<const N: usize> { trees: Vec<AllNodeTypes<N>> }
enum AllNodeTypes<const N: usize> { Inner(Box<InnerNode<N>>), Leaf(Box<LeafNode<N>>) }
struct LeafNode<const N: usize> { values: Vec<Vector<N>> }
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: AllNodeTypes<N>,
    right_node: AllNodeTypes<N>,
}

fn build_a_tree<const N: usize>(max_size_of_node: i32, vectors: &Vec<Vector<N>>) -> AllNodeTypes<N> {
    // If we have very few points, return a leaf node
    if vectors.len() <= (max_size_of_node as usize) {
        return AllNodeTypes::Leaf(Box::new(LeafNode::<N> { values: vectors.clone() }));
    }
    // Otherwise, build an inner node, and recursively build left and right
    let (hyperplane, above, below) = build_hyperplane_bwn_two_random_points::<N>(vectors);
    let node_above = build_a_tree::<N>(max_size_of_node, &above);
    let node_below = build_a_tree::<N>(max_size_of_node, &below);
    return AllNodeTypes::Inner(Box::new(InnerNode::<N> {
        hyperplane: hyperplane, left_node: node_below, right_node: node_above}));
}

fn deduplicate_vector_list<const N: usize>(vectors: &Vec<Vector<N>>) -> Vec<Vector<N>> {
    let mut deduplicated_list: Vec<Vector<N>> = Vec::new();
    let mut hashes_seen: std::collections::HashSet<HashKey<N>> = std::collections::HashSet::new();
    for &vector in vectors.iter() {
        let hash_key = vector.to_hashkey();
        if !hashes_seen.contains(&hash_key) {
            hashes_seen.insert(hash_key);
            deduplicated_list.push(vector);
        }
    }
    return deduplicated_list;
}

fn build_an_index<const N: usize>(num_trees: i32, max_size_of_node: i32, vectors: &Vec<Vector<N>>) -> ANNIndex<N> {
    let unique_vectors = deduplicate_vector_list(vectors);
    let mut trees: Vec<AllNodeTypes<N>> = Vec::new();
    for _ in 0..num_trees {
        trees.push(build_a_tree::<N>(max_size_of_node, &unique_vectors))
    }
    return ANNIndex::<N> { trees: trees };
}

fn get_candidates_per_tree<const N: usize>(vector: Vector<N>, num_candidates: i32, tree: &AllNodeTypes<N>, global_list: &mut Vec<Vector<N>>) -> i32 {
    // We take everything in the leaf node we end up with. If we still need candidates, we take closer ones from the alternate subtree
    match tree {
        AllNodeTypes::Leaf(box_leaf) => {
            let leaf_values = &(*box_leaf).values;
            let num_candidates_found = std::cmp::min(num_candidates as usize, leaf_values.len());
            for i in 0..num_candidates_found {
                global_list.push(leaf_values[i])
            }
            return num_candidates_found as i32;
        }
        AllNodeTypes::Inner(box_inner) => {
            let (correct_tree, backup_tree) = if (*box_inner).hyperplane.point_is_above(&vector) {
                (&(*box_inner).right_node, &(*box_inner).left_node)
            } else {
                (&(*box_inner).left_node, &(*box_inner).right_node)
            };
            let mut fetched = get_candidates_per_tree::<N>(vector, num_candidates, correct_tree, global_list);
            if fetched < num_candidates {
                fetched += get_candidates_per_tree::<N>(vector, num_candidates - fetched, backup_tree, global_list);
            };
            return fetched;
        }
    }
}

fn search_on_index<const N: usize>(vector: Vector<N>, top_k: i32, index: &ANNIndex<N>) -> Vec<Vector<N>> {
    // Get top_k items per tree, deduplicate them, rank them by Euc distance and return the overall top_k
    let mut initial_global_list: Vec<Vector<N>> = Vec::new();
    for tree in index.trees.iter() {
        get_candidates_per_tree::<N>(vector, top_k, tree, &mut initial_global_list);
    }
    let unique_candidate_list = deduplicate_vector_list(&initial_global_list);
    let enumerated_iter = unique_candidate_list.iter().enumerate();
    let mut idx_sq_euc_dis: Vec<(usize, f32)> = enumerated_iter.map(|(i, can)| (i, can.sq_euc_dis(&vector))).collect();
    idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut final_candidates: Vec<Vector<N>> = Vec::new();
    let num_candidates_to_return = std::cmp::min(top_k as usize, unique_candidate_list.len());
    for i in 0..num_candidates_to_return {
        final_candidates.push(unique_candidate_list[idx_sq_euc_dis[i].0]);
    }
    return final_candidates;
}

fn main() {
    const DIM: usize = 30;
    const NUM_VECTORS: i32 = 1000000;
    const NUM_TREES: i32 = 3;
    const TOP_K: i32 = 20;
    const MAX_NODE_SIZE : i32 = 15;
    // Generate the data
    let start = std::time::Instant::now();
    let mut rng = rand::thread_rng();
    let my_input_data: Vec<Vector<DIM>> = (1..NUM_VECTORS).map(
        |_x| Vector {coordinates: rng.gen::<[f32; DIM]>()}).collect();
    let duration = start.elapsed();
    let vector = Vector {coordinates: rng.gen::<[f32; DIM]>()};
    println!("Generated {} vectors in {}-D in {:?}", NUM_VECTORS, DIM, duration);
    // Try the naive exact-search for TOP_K elements - if TOP_K << log(NUM_VECTORS) then
    // this runs in O(TOP_K * DIM * NUM_VECTORS) else O(DIM * NUM_VECTORS * log(NUM_VECTORS))
    let start = std::time::Instant::now();
    let enumerated_iter = my_input_data.iter().enumerate();
    let mut idx_sq_euc_dis: Vec<(usize, f32)> = enumerated_iter.map(|(i, can)| (i, can.sq_euc_dis(&vector))).collect();
    idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut final_candidates_brute: Vec<Vector<DIM>> = Vec::new();
    for i in 0..TOP_K as usize {
        final_candidates_brute.push(my_input_data[idx_sq_euc_dis[i].0])
    }
    let duration = start.elapsed();
    println!("Found {} vectors via brute-search in {}-D in {:?}", final_candidates_brute.len(), DIM, duration);
    // Build the ANN index
    let index = build_an_index::<DIM>(NUM_TREES, MAX_NODE_SIZE, &my_input_data);
    // Perform ANN search
    let start = std::time::Instant::now();
    let search_results = search_on_index::<DIM>(vector, TOP_K, &index);
    let duration = start.elapsed();
    println!("Found {} vectors via ANN-search in {}-D in {:?}", search_results.len(), DIM, duration);
}
