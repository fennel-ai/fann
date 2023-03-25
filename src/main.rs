use rand::{prelude::SliceRandom, Rng};

#[derive(Eq, PartialEq, Hash)]
struct HashKey<const N: usize>([u32; N]);

#[derive(Copy, Clone)]
struct Vector<const N: usize> ([f32; N]);
impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped_iter = self.0.iter().zip(vector.0).map(|(a, b)| b - a);
        let coordinates: [f32; N] = mapped_iter.collect::<Vec<_>>().try_into().unwrap();
        return Vector(coordinates);
    }
    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped_iter = self.0.iter().zip(vector.0).map(|(a, b)| (a + b) / 2.0);
        let coordinates: [f32; N] = mapped_iter.collect::<Vec<_>>().try_into().unwrap();
        return Vector(coordinates);
    }
    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.0.iter().zip(vector.0);
        return zipped_iter.map(|(a, b)| a * b).sum::<f32>();
    }
    pub fn to_hashkey(&self) -> HashKey<N> {
        // f32 in Rust doesn't implement hash - we use byte representation to deduplicate which is unsafe in that it
        // cannot differentiate ~16 mil representations of NaN under IEEE-754 but safe for our demonstration
        let bit_iter = self.0.iter().map(|a| a.to_bits());
        let u32_data: [u32; N] = bit_iter.collect::<Vec<_>>().try_into().unwrap();
        return HashKey::<N>(u32_data);
    }
    pub fn sq_euc_dis(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.0.iter().zip(vector.0);
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

enum Node<const N: usize> { Inner(Box<InnerNode<N>>), Leaf(Box<LeafNode<N>>) }
struct LeafNode<const N: usize>(Vec<usize>);
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}
struct ANNIndex<const N: usize> { trees: Vec<Node<N>>, values: Vec<Vector<N>> }
impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane_bwn_two_random_points(ids_of_interest: &Vec<usize>, all_vectors: &Vec<Vector<N>>) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<_> = ids_of_interest.choose_multiple(&mut rand::thread_rng(), 2).collect();
        // We use implicit Cartesian equation for hyperplane n * (x - x_0) = 0. n (normal vector) is the coefs x_1 to x_n
        let coefficients = all_vectors[*sample[0]].subtract_from(&all_vectors[*sample[1]]);
        let point_on_plane = all_vectors[*sample[0]].avg(&all_vectors[*sample[1]]);
        let constant = - coefficients.dot_product(&point_on_plane);
        // Figure out which points lie above and below
        let mut above: Vec<usize> = Vec::new();
        let mut below: Vec<usize> = Vec::new();
        let hyperplane = HyperPlane::<N> { coefficients: coefficients, constant: constant };
        for &id in ids_of_interest.iter() {
            if hyperplane.point_is_above(&all_vectors[id]) { above.push(id) } else { below.push(id) };
        }
        return (hyperplane, above, below);
    }

    fn build_a_tree(max_size_of_node: i32, ids_of_interest: &Vec<usize>, all_vectors: &Vec<Vector<N>>) -> Node<N> {
        // If we have very few ids of interest, return a leaf node
        if ids_of_interest.len() <= (max_size_of_node as usize) {
            return Node::Leaf(Box::new(LeafNode::<N>(ids_of_interest.clone())));
        }
        // Otherwise, build an inner node, and recursively build left and right
        let (hyperplane, above, below) = Self::build_hyperplane_bwn_two_random_points(ids_of_interest, all_vectors);
        let node_above = Self::build_a_tree(max_size_of_node, &above, all_vectors);
        let node_below = Self::build_a_tree(max_size_of_node, &below, all_vectors);
        return Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: hyperplane, left_node: node_below, right_node: node_above}));
    }

    fn deduplicate_vector_list(vectors: &Vec<Vector<N>>) -> Vec<Vector<N>> {
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

    pub fn build_an_index(num_trees: i32, max_size_of_node: i32, vectors: &Vec<Vector<N>>) -> ANNIndex<N> {
        let unique_vectors = Self::deduplicate_vector_list(vectors);
        let all_ids: Vec<usize> = (0..unique_vectors.len()).collect();
        let mut trees: Vec<Node<N>> = Vec::new();
        for _ in 0..num_trees {
            trees.push(Self::build_a_tree(max_size_of_node, &all_ids, &unique_vectors))
        }
        return ANNIndex::<N> { trees: trees, values: unique_vectors };
    }

    fn get_candidates_per_tree(vector: Vector<N>, num_candidates: i32, tree: &Node<N>, candidates: &mut std::collections::HashSet<usize>) -> i32 {
        // We take everything in the leaf node we end up with. If we still need candidates, we take closer ones from the alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(**box_leaf).0;
                let num_candidates_found = std::cmp::min(num_candidates as usize, leaf_values.len());
                for i in 0..num_candidates_found {
                    candidates.insert(leaf_values[i]);
                }
                return num_candidates_found as i32;
            }
            Node::Inner(box_inner) => {
                let (correct_tree, backup_tree) = if (*box_inner).hyperplane.point_is_above(&vector) {
                    (&(*box_inner).right_node, &(*box_inner).left_node)
                } else {
                    (&(*box_inner).left_node, &(*box_inner).right_node)
                };
                let mut fetched = Self::get_candidates_per_tree(vector, num_candidates, correct_tree, candidates);
                if fetched < num_candidates {
                    fetched += Self::get_candidates_per_tree(vector, num_candidates - fetched, backup_tree, candidates);
                };
                return fetched;
            }
        }
    }

    pub fn search_on_index(&self, vector: Vector<N>, top_k: i32) -> Vec<Vector<N>> {
        // Get top_k items per tree, deduplicate them, rank them by Euc distance and return the overall top_k
        // Get the identifiers for the candidate vectors (identifier is the index into self.values)
        let mut candidates: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for tree in self.trees.iter() {
            Self::get_candidates_per_tree(vector, top_k, tree, &mut candidates);
        }
        let mut idx_sq_euc_dis: Vec<(usize, f32)> = candidates.iter().map(|&idx| (idx, self.values[idx].sq_euc_dis(&vector))).collect();
        idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut final_candidates: Vec<Vector<N>> = Vec::new();
        let num_candidates_to_return = std::cmp::min(top_k as usize, candidates.len());
        for i in 0..num_candidates_to_return {
            final_candidates.push(self.values[idx_sq_euc_dis[i].0]);
        }
        return final_candidates;
    }
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
    let my_input_data: Vec<Vector<DIM>> = (1..NUM_VECTORS).map(|_x| Vector(rng.gen::<[f32; DIM]>())).collect();
    let duration = start.elapsed();
    let vector = Vector(rng.gen::<[f32; DIM]>());
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
    let index = ANNIndex::<DIM>::build_an_index(NUM_TREES, MAX_NODE_SIZE, &my_input_data);
    // Perform ANN search
    let start = std::time::Instant::now();
    let search_results = index.search_on_index(vector, TOP_K);
    let duration = start.elapsed();
    println!("Found {} vectors via ANN-search in {}-D in {:?}", search_results.len(), DIM, duration);
    // Perform a batch of ANN searches
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let vector = Vector(rng.gen::<[f32; DIM]>());
        index.search_on_index(vector, TOP_K);
    }
    let duration = start.elapsed() / 100;
    println!("Bulk ANN-search in {}-D has average time {:?}", DIM, duration);
}
