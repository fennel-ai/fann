use rand::{prelude::SliceRandom, Rng};

struct HyperPlane<const N: usize> { coefficients: [f32; N], constant: f32 }
impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &[f32; N]) -> bool { self.coefficients.iter().zip(point).map(|(a, b)| a * b).sum::<f32>() + self.constant >= 0.0 }
}

fn build_hyperplane_bwn_two_random_points<const N: usize>(all_points: &Vec<[f32; N]>) -> (HyperPlane<N>, Vec<[f32; N]>,  Vec<[f32; N]>) {
    let sample: Vec<_> = all_points.choose_multiple(&mut rand::thread_rng(), 2).collect();
    // We use implicit Cartesian equation for hyperplane n * (x - x_0) = 0. n (normal vector) is the coefs x_1 to x_n
    let coefficients: [f32; N] = sample[0].iter().zip(sample[1]).map(|(a, b)| b - a).collect::<Vec<_>>().try_into().unwrap();
    let point_on_plane: [f32; N] = sample[0].iter().zip(sample[1]).map(|(a, b)| (a + b) / 2.0).collect::<Vec<_>>().try_into().unwrap();
    let constant =  - coefficients.iter().zip(point_on_plane).map(|(a, b)| a * b).sum::<f32>();
    // Figure out which points lie above and below
    let mut above: Vec<[f32; N]> = Vec::new();
    let mut below: Vec<[f32; N]> = Vec::new(); 
    let hyperplane = HyperPlane::<N>{coefficients: coefficients, constant: constant};
    for &point in all_points.iter() {
        if hyperplane.point_is_above(&point) { above.push(point) } else { below.push(point) };
    }
    return (hyperplane, above, below);
}

enum AllNodeTypes<const N: usize> { Inner(Box<InnerNode<N>>), Leaf(Box<LeafNode<N>>) }
struct InnerNode<const N: usize> { hyperplane: HyperPlane<N>, left_node: AllNodeTypes<N>, right_node: AllNodeTypes<N> }
struct LeafNode<const N: usize> { values: Vec<[f32; N]> }

fn build_a_tree<const N: usize>(max_size_of_node: usize, points: &Vec<[f32; N]>) -> AllNodeTypes<N> {
    // If we have very few points, return a leaf node
    if points.len() <= max_size_of_node { return AllNodeTypes::Leaf(Box::new(LeafNode::<N>{ values: points.clone() })) }
    // Otherwise, build an inner node, and recursively build left and right
    let (hyperplane, above, below) = build_hyperplane_bwn_two_random_points::<N>(points);
    let node_above = build_a_tree::<N>(max_size_of_node, &above);
    let node_below = build_a_tree::<N>(max_size_of_node, &below);
    return AllNodeTypes::Inner(Box::new(InnerNode::<N>{ hyperplane: hyperplane, left_node: node_below, right_node: node_above }));
}

#[derive(Eq, PartialEq, Hash)]
struct HashKey<const N: usize>([u32; N]);

fn deduplicate_point_list<const N: usize>(points: &Vec<[f32; N]>) -> Vec<[f32; N]> {
    // f32 in Rust doesn't implement hash - we use byte representation to deduplicate which is unsafe in that it
    // cannot differentiate ~16 mil representations of NaN under IEEE-754 but safe for our demonstration
    let mut deduplicated_list: Vec<[f32; N]> = Vec::new();
    let mut point_hashes_seen: std::collections::HashSet<HashKey<N>> = std::collections::HashSet::new();
    for &point in points.iter() {
        let hash_key = HashKey(point.iter().map(|a| a.to_bits()).collect::<Vec<_>>().try_into().unwrap());
        if !point_hashes_seen.contains(&hash_key) { 
            point_hashes_seen.insert(hash_key); 
            deduplicated_list.push(point); 
        }
    }
    return deduplicated_list
}

struct ANNIndex<const N: usize> { trees: Vec<AllNodeTypes<N>>, num_items_in_index: usize }

fn build_an_index<const N: usize>(num_trees: i32, max_size_of_node: usize, points: &Vec<[f32; N]>) -> ANNIndex<N> {
    // points are expected to be unique, else we can simply take a set / deduplicate here
    let unique_points = deduplicate_point_list(points);
    let mut trees: Vec<AllNodeTypes<N>> = Vec::new();
    for _ in 0..num_trees { trees.push(build_a_tree::<N>(max_size_of_node, &unique_points)) }
    return ANNIndex::<N>{ trees: trees, num_items_in_index: unique_points.len() }
}

fn get_candidates_per_tree<const N: usize>(point: [f32; N], num_candidates: i32, tree: &AllNodeTypes<N>, global_list: &mut Vec<[f32; N]>) -> i32 {
    // We take everything in the leaf node we end up with. If we still need candidates, we take closer ones from the alternate subtree
    match tree {
        AllNodeTypes::Leaf(box_leaf) => {
            let leaf_values = &(*box_leaf).values;
            let num_candidates_found = std::cmp::min(num_candidates as usize, leaf_values.len());
            for i in 0..num_candidates_found { global_list.push(leaf_values[i]) }
            return num_candidates_found as i32 }
        AllNodeTypes::Inner(box_inner) => {
            let (correct_tree, backup_tree) = if (*box_inner).hyperplane.point_is_above(&point) { 
                (&(*box_inner).right_node, &(*box_inner).left_node) 
            } else { 
                (&(*box_inner).left_node, &(*box_inner).right_node) 
            };
            let mut num_fetched = get_candidates_per_tree::<N>(point, num_candidates, correct_tree, global_list);
            if num_fetched < num_candidates { 
                num_fetched += get_candidates_per_tree::<N>(point, num_candidates - num_fetched, backup_tree, global_list) 
            };
            return num_fetched }
    }
}

fn search_on_index<const N: usize>(point: [f32; N], top_k: i32, index: ANNIndex<N>) -> Vec<[f32; N]> {
    // Get top_k items per tree, deduplicate them, rank them by Euc distance and return the overall top_k
    let mut initial_global_list: Vec<[f32; N]> = Vec::new();
    for tree in index.trees.iter() { get_candidates_per_tree::<N>(point, top_k, tree, &mut initial_global_list); }
    let unique_candidate_list = deduplicate_point_list(&initial_global_list);
    let mut sq_euc_distance_to_point: Vec<(usize, f32)> = unique_candidate_list.iter().enumerate().map(|(i, can)| (i, can.iter().zip(point).map(|(a, b)| (a - b).powi(2)).sum())).collect();
    sq_euc_distance_to_point.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut final_candidates: Vec<[f32; N]> = Vec::new();
    let num_candidates_to_return = std::cmp::min(top_k as usize, unique_candidate_list.len());
    for i in 0..num_candidates_to_return { final_candidates.push(unique_candidate_list[sq_euc_distance_to_point[i].0]); }
    return final_candidates;
}

fn main() {
    const DIM: usize = 10;
    const NUM_VECTORS: i32 = 1000000;
    const NUM_TREES: i32 = 3;
    const TOP_K = 15;
    // Generate the data
    let start = std::time::Instant::now();
    let mut rng = rand::thread_rng();
    let my_input_data: Vec<[f32; DIM]> = (1..NUM_VECTORS).map(|x| rng.gen::<[f32; DIM]>()).collect();
    let duration = start.elapsed();
    println!("Generated {} vectors in {}-D in {:?}", NUM_VECTORS, DIM, duration);
    // Try the naive exact-search for TOP_K elements - if TOP_K << log(NUM_VECTORS) then
    // this runs in O(TOP_K * DIM * NUM_VECTORS) else O(DIM * NUM_VECTORS * log(NUM_VECTORS))
    // Build the ANN index
    // Perform ANN search

    // println!("Hello, world!");
    // let points: Vec<[f32; N]> = vec![[3.0,4.0],[5.0,6.0],[7.0,9.0],[5.0,2.0],[1.0,3.0],[7.0,8.0]];
    // let index = build_an_index::<N>(3, 2, &points);
    // let global_list = search_on_index::<N>([7.0, 8.5], 3, index);
    // println!("{:?}", global_list);
}
