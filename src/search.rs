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
pub struct Vector<const N: usize>(pub [f32; N]);
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
        // f32 in Rust doesn't implement hash - we use byte to deduplicate which is unsafe in that it
        // cannot differentiate ~16 mil ways NaN is written under IEEE-754 but safe for us
        let bit_iter = self.0.iter().map(|a| a.to_bits());
        let u32_data: [u32; N] = bit_iter.collect::<Vec<_>>().try_into().unwrap();
        return HashKey::<N>(u32_data);
    }
    pub fn sq_euc_dis(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.0.iter().zip(vector.0);
        return zipped_iter.map(|(a, b)| (a - b).powi(2)).sum();
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}
struct LeafNode<const N: usize>(Vec<usize>);
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}
pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<i32>,
    values: Vec<Vector<N>>,
}
impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &Vec<usize>,
        all_vectors: &Vec<Vector<N>>,
    ) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<_> = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect();
        // implicit Cartesian eq for hyperplane n * (x - x_0) = 0. n (normal vector) is the coefs x_1 to x_n
        let coefficients = all_vectors[*sample[0]].subtract_from(&all_vectors[*sample[1]]);
        let point_on_plane = all_vectors[*sample[0]].avg(&all_vectors[*sample[1]]);
        let constant = -coefficients.dot_product(&point_on_plane);
        let (mut above, mut below) = (Vec::new(), Vec::new());
        let hyperplane = HyperPlane::<N> {
            coefficients: coefficients,
            constant: constant,
        };
        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vectors[id]) {
                above.push(id)
            } else {
                below.push(id)
            };
        }
        return (hyperplane, above, below);
    }

    fn build_a_tree(max_size: i32, indexes: &Vec<usize>, all_vecs: &Vec<Vector<N>>) -> Node<N> {
        if indexes.len() <= (max_size as usize) {
            return Node::Leaf(Box::new(LeafNode::<N>(indexes.clone())));
        }
        let (hyperplane, above, below) = Self::build_hyperplane(indexes, all_vecs);
        let node_above = Self::build_a_tree(max_size, &above, all_vecs);
        let node_below = Self::build_a_tree(max_size, &below, all_vecs);
        return Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: hyperplane,
            left_node: node_below,
            right_node: node_above,
        }));
    }

    fn deduplicate(
        vectors: &Vec<Vector<N>>,
        ids: &Vec<i32>,
        dedup_vectors: &mut Vec<Vector<N>>,
        ids_of_dedup_vectors: &mut Vec<i32>,
    ) {
        let mut hashes_seen = std::collections::HashSet::new();
        for i in 1..vectors.len() {
            let hash_key = vectors[i].to_hashkey();
            if !hashes_seen.contains(&hash_key) {
                hashes_seen.insert(hash_key);
                dedup_vectors.push(vectors[i]);
                ids_of_dedup_vectors.push(ids[i]);
            }
        }
    }

    pub fn build_index(
        num_trees: i32,
        max_size: i32,
        vecs: &Vec<Vector<N>>,
        vec_ids: &Vec<i32>,
    ) -> ANNIndex<N> {
        let (mut unique_vecs, mut ids, mut trees) = (Vec::new(), Vec::new(), Vec::new());
        Self::deduplicate(vecs, vec_ids, &mut unique_vecs, &mut ids);
        // Trees hold an index into the [unique_vecs] list which is not necessarily its id, if duplicates existed
        let all_indexes: Vec<usize> = (0..unique_vecs.len()).collect();
        for _ in 0..num_trees {
            let tree = Self::build_a_tree(max_size, &all_indexes, &unique_vecs);
            trees.push(tree);
        }
        return ANNIndex::<N> {
            trees: trees,
            ids: ids,
            values: unique_vecs,
        };
    }

    fn tree_result(
        vector: Vector<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &mut std::collections::HashSet<usize>,
    ) -> i32 {
        // We take everything in the leaf node. If we still need, we take ones from the alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = std::cmp::min(n as usize, leaf_values.len());
                for i in 0..num_candidates_found {
                    candidates.insert(leaf_values[i]);
                }
                return num_candidates_found as i32;
            }
            Node::Inner(box_inner) => {
                let (correct_tree, backup_tree) = if (*box_inner).hyperplane.point_is_above(&vector)
                {
                    (&(box_inner.right_node), &(box_inner.left_node))
                } else {
                    (&(box_inner.left_node), &(box_inner.right_node))
                };
                let mut fetched = Self::tree_result(vector, n, correct_tree, candidates);
                if fetched < n {
                    fetched += Self::tree_result(vector, n - fetched, backup_tree, candidates);
                };
                return fetched;
            }
        }
    }

    pub fn search_approximate(&self, vector: Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let mut candidates = std::collections::HashSet::new();
        for tree in self.trees.iter() {
            Self::tree_result(vector, top_k, tree, &mut candidates);
        }
        let mut idx_sq_euc_dis: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&idx| (idx, self.values[idx].sq_euc_dis(&vector)))
            .collect();
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
