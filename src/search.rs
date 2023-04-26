use rand::prelude::SliceRandom;

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}
impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }
}

#[derive(Copy, Clone)]
pub struct Vector<const N: usize> (pub [f32; N]);
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


enum Node<const N: usize> { Inner(Box<InnerNode<N>>), Leaf(Box<LeafNode<N>>) }
struct LeafNode<const N: usize>(Vec<usize>);
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}
pub struct ANNIndex<const N: usize> { trees: Vec<Node<N>>, ids: Vec<i32>, values: Vec<Vector<N>> }
impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane_bwn_two_random_points(indexes_of_interest: &Vec<usize>, all_vectors: &Vec<Vector<N>>) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<_> = indexes_of_interest.choose_multiple(&mut rand::thread_rng(), 2).collect();
        // implicit Cartesian eq for hyperplane n * (x - x_0) = 0. n (normal vector) is the coefs x_1 to x_n
        let coefficients = all_vectors[*sample[0]].subtract_from(&all_vectors[*sample[1]]);
        let point_on_plane = all_vectors[*sample[0]].avg(&all_vectors[*sample[1]]);
        let constant = - coefficients.dot_product(&point_on_plane);
        // Figure out which points lie above and below
        let mut above: Vec<usize> = Vec::new();
        let mut below: Vec<usize> = Vec::new();
        let hyperplane = HyperPlane::<N> { coefficients: coefficients, constant: constant };
        for &id in indexes_of_interest.iter() {
            if hyperplane.point_is_above(&all_vectors[id]) { above.push(id) } else { below.push(id) };
        }
        return (hyperplane, above, below);
    }

    fn build_a_tree(max_size_of_node: i32, indexes_of_interest: &Vec<usize>, all_vectors: &Vec<Vector<N>>) -> Node<N> {
        if indexes_of_interest.len() <= (max_size_of_node as usize) {
            return Node::Leaf(Box::new(LeafNode::<N>(indexes_of_interest.clone())));
        }
        // Otherwise, build an inner node, and recursively build left and right
        let (hyperplane, above, below) = Self::build_hyperplane_bwn_two_random_points(indexes_of_interest, all_vectors);
        let node_above = Self::build_a_tree(max_size_of_node, &above, all_vectors);
        let node_below = Self::build_a_tree(max_size_of_node, &below, all_vectors);
        return Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: hyperplane, left_node: node_below, right_node: node_above}));
    }

    fn deduplicate_vector_list(
        vectors: &Vec<Vector<N>>, ids: &Vec<i32>, dedup_vectors: &mut Vec<Vector<N>>, 
        ids_of_dedup_vectors: &mut Vec<i32>) {
        let mut hashes_seen: std::collections::HashSet<HashKey<N>> = std::collections::HashSet::new();
        for i in 1..vectors.len() {
            let hash_key = vectors[i].to_hashkey();
            if !hashes_seen.contains(&hash_key) {
                hashes_seen.insert(hash_key);
                dedup_vectors.push(vectors[i]);
                ids_of_dedup_vectors.push(ids[i]);
            }
        }
    }

    pub fn build_an_index(num_trees: i32, max_size_of_node: i32, vectors: &Vec<Vector<N>>, vector_ids: &Vec<i32>) -> ANNIndex<N> {
        let mut unique_vectors = Vec::new();
        let mut ids_of_unique_vectors = Vec::new();
        Self::deduplicate_vector_list(vectors, vector_ids, &mut unique_vectors, &mut ids_of_unique_vectors);
        // Trees hold an index into the [unique_vectors] list which is not necessarily its id, if duplicates existed
        let all_indexes_unique_vectors: Vec<usize> = (0..unique_vectors.len()).collect();
        let mut trees: Vec<Node<N>> = Vec::new();
        for _ in 0..num_trees {
            trees.push(Self::build_a_tree(max_size_of_node, &all_indexes_unique_vectors, &unique_vectors))
        }
        return ANNIndex::<N> { trees: trees, ids: ids_of_unique_vectors, values: unique_vectors };
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

    pub fn search_approximate(&self, vector: Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let mut candidates: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for tree in self.trees.iter() {
            Self::get_candidates_per_tree(vector, top_k, tree, &mut candidates);
        }
        let mut idx_sq_euc_dis: Vec<(usize, f32)> = candidates.iter().map(|&idx| (idx, self.values[idx].sq_euc_dis(&vector))).collect();
        idx_sq_euc_dis.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut final_candidates: Vec<(i32, f32)> = Vec::new();
        let num_candidates_to_return = std::cmp::min(top_k as usize, candidates.len());
        for i in 0..num_candidates_to_return {
            let (index, distance) = idx_sq_euc_dis[i];
            final_candidates.push((self.ids[index], distance));
        }
        return final_candidates;
    }
}
