pub mod test_common;

mod partition;
mod bounds;

use self::bounds::*;

use num_traits::Float;
use core::cmp;

pub trait KdTreePoint<F: Float>: Copy + PartialEq {
    fn dist_1d(left: F, right: F, _dim: usize) -> F {
        let diff = left - right;

        diff * diff
    }

    fn dims(&self) -> usize;
    fn dim(&self, i: usize) -> F;
    fn dist(&self, other: &Self) -> F {
        let mut sum = F::zero();

        for i in 0..self.dims() {
            let x = self.dim(i);
            let y = other.dim(i);
            let diff = x - y;

            sum = sum + diff * diff;
        }

        sum
    }

    #[inline]
    fn to_vec(&self) -> Vec<F> {
        (0..self.dims())
            .map(|x| self.dim(x))
            .collect()
    } 
}

pub struct NearestNeighboursIter<'a, F: Float, T> {
    range: F,
    kdtree: &'a KdTree<F, T>,
    ref_node: T,
    node_stack: Vec<usize>,
}

impl<'a, F: Float, T> Iterator for NearestNeighboursIter<'a, F, T>
    where T: KdTreePoint<F>
{
    type Item = (F, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let p = &self.ref_node;

        loop {
            let node_idx = self.node_stack.pop()?;
            let node = &self.kdtree.nodes[node_idx];

            let splitting_value = node.split_on;
            let point_splitting_dim_value = p.dim(node.dimension);
            let distance_on_single_dimension = T::dist_1d(splitting_value, point_splitting_dim_value, node.dimension);

            if distance_on_single_dimension <= self.range {
                if let Some(idx) = node.left_node {
                    self.node_stack.push(idx);
                }

                if let Some(idx) = node.right_node {
                    self.node_stack.push(idx);
                }

                let dist = p.dist(&node.point);
                if dist <= self.range {
                    return Some((dist, &node.point));
                }
            } else if point_splitting_dim_value <= splitting_value {
                if let Some(idx) = node.left_node {
                    self.node_stack.push(idx);
                }
            } else {
                if let Some(idx) = node.right_node {
                    self.node_stack.push(idx);
                }
            }
        }
    }
}

pub struct KdTree<F: Float, KP> {
    nodes: Vec<KdTreeNode<F, KP>>,

    node_adding_dimension: usize,
    node_depth_during_last_rebuild: usize,
    current_node_depth: usize,
}

impl<F: Float, KP: KdTreePoint<F>> KdTree<F, KP> {
    #[inline]
    pub fn empty() -> Self {
        KdTree {
            nodes: vec![],
            node_adding_dimension: 0,
            node_depth_during_last_rebuild: 0,
            current_node_depth: 0,
        }
    }

    pub fn new(mut points: &mut [KP]) -> Self {
        if points.len() == 0 {
            panic!("empty vector point not allowed");
        }

        let mut tree = Self::empty();

        tree.rebuild_tree(&mut points);

        tree
    }

    pub fn rebuild_tree(&mut self, points: &mut [KP]) {
        self.nodes.clear();
        self.nodes.reserve(points.len());

        self.node_depth_during_last_rebuild = 0;
        self.current_node_depth = 0;

        let rect = Bounds::new_from_points(points);
        self.build_tree(points, &rect, 1);
    }

    /// Can be used if you are sure that the tree is degenerated or if you will never again insert the nodes into the tree.
    pub fn gather_points_and_rebuild(&mut self) {
        let original = core::mem::replace(self, Self::empty());
        let mut points: Vec<_> = original.into_iter().collect();

        self.rebuild_tree(&mut points);
    }

    pub fn nearest_search(&self, node: &KP) -> (F, &KP) {
        let mut nearest_neighbor = 0usize;
        let mut best_distance = self.nodes[0].point.dist(&node);

        self.nearest_search_impl(node, 0usize, &mut best_distance, &mut nearest_neighbor);

        (best_distance, &self.nodes[nearest_neighbor].point)
    }

    pub fn nearest_search_dist(&self, node: KP, dist: F) -> NearestNeighboursIter<'_, F, KP> {
        let mut node_stack = Vec::with_capacity(16);
        node_stack.push(0);

        NearestNeighboursIter {
            range: dist,
            kdtree: self,
            ref_node: node,
            node_stack,
        }
    }

    #[inline]
    pub fn has_neighbor_in_range(&self, node: &KP, range: F) -> bool {
        let squared_range = range * range;

        self.distance_squared_to_nearest(node) <= squared_range
    }

    #[inline]
    pub fn distance_squared_to_nearest(&self, node: &KP) -> F {
        self.nearest_search(node).0
    }

    pub fn insert_nodes_and_rebuild<I: Iterator<Item = KP>>(&mut self, nodes_to_add: I) {
        let original = std::mem::replace(self, Self::empty());
        let mut points: Vec<_> = original
            .into_iter()
            .chain(nodes_to_add)
            .collect();

        self.rebuild_tree(&mut points);
    }

    pub fn insert_node(&mut self, node_to_add: KP) {
        let mut current_index = 0;
        let dimension = self.node_adding_dimension;
        let dims = node_to_add.to_vec();
        let index_of_new_node = self.add_node(node_to_add, dimension,dims[dimension]);

        self.node_adding_dimension = (dimension + 1) % dims.len();
        let mut should_pop_node = false;

        let mut depth = 0;
        loop {
            depth +=1 ;

            let nodes = &mut self.nodes[0 ..= current_index];

            let current_node_dimension = nodes[current_index].dimension;
            let current_node_split_on = nodes[current_index].split_on;
            let current_node_left_node = nodes[current_index].left_node;
            let current_node_right_node = nodes[current_index].right_node;

            if dims[current_node_dimension] <= current_node_split_on {
                if let Some(left_node_index) = current_node_left_node {
                    current_index = left_node_index
                } else {
                    if self.nodes[current_index].point.eq(&self.nodes[index_of_new_node].point) {
                        should_pop_node = true;
                    } else {
                        self.nodes[current_index].left_node = Some(index_of_new_node);
                    }
                    break;
                }
            } else {
                if let Some(right_node_index) = current_node_right_node {
                    current_index = right_node_index
                } else {
                    if self.nodes[current_index].point.eq(&self.nodes[index_of_new_node].point) {
                        should_pop_node = true;
                    } else {
                        self.nodes[current_index].right_node = Some(index_of_new_node);
                    }
                    break;
                }
            }
        }

        if should_pop_node {
            self.nodes.pop();
        }

        if F::from(self.node_depth_during_last_rebuild).unwrap() * F::from(4.0).unwrap() < F::from(depth).unwrap()  {
            self.gather_points_and_rebuild();
        }
    }

    fn nearest_search_impl(&self, p: &KP, searched_index: usize, best_distance_squared: &mut F, best_leaf_found: &mut usize) {
        let node = &self.nodes[searched_index];

        let splitting_value = node.split_on;
        let point_splitting_dim_value = p.dim(node.dimension);

        let (closer_node, farther_node) = if point_splitting_dim_value <= splitting_value {
            (node.left_node, node.right_node)
        } else {
            (node.right_node, node.left_node)
        };

        if let Some(closer_node) = closer_node {
            self.nearest_search_impl(p, closer_node, best_distance_squared, best_leaf_found);
        }

        let distance = p.dist(&node.point);
        if distance < *best_distance_squared {
            *best_distance_squared = distance;
            *best_leaf_found = searched_index;
        }

        if let Some(farther_node) = farther_node {
            let distance_on_single_dimension = KP::dist_1d(splitting_value, point_splitting_dim_value, node.dimension);

            if distance_on_single_dimension <= *best_distance_squared {
                self.nearest_search_impl(p, farther_node, best_distance_squared, best_leaf_found);
            }
        }
    }

    fn add_node(&mut self, p: KP, dimension: usize, split_on: F) -> usize {
        let node = KdTreeNode::new(p, dimension, split_on);

        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn build_tree(&mut self, nodes: &mut [KP], bounds: &Bounds<F>, depth : usize) -> usize {
        let splitting_index = partition::partition_sliding_midpoint(nodes, bounds.get_midvalue_of_widest_dim(), bounds.get_widest_dim());
        let pivot_value = nodes[splitting_index].dim(bounds.get_widest_dim());

        let node_id = self.add_node(nodes[splitting_index], bounds.get_widest_dim(), pivot_value);
        let nodes_len = nodes.len();

        if splitting_index > 0 {
            let left_rect = bounds.clone_moving_max(pivot_value, bounds.get_widest_dim());
            let left_child_id = self.build_tree(&mut nodes[0 .. splitting_index], &left_rect, depth+1);
            self.nodes[node_id].left_node = Some(left_child_id);
        }

        if splitting_index < nodes.len() - 1 {
            let right_rect = bounds.clone_moving_min(pivot_value, bounds.get_widest_dim());

            let right_child_id = self.build_tree(&mut nodes[splitting_index + 1..nodes_len], &right_rect, depth+1);
            self.nodes[node_id].right_node = Some(right_child_id);
        }

        self.node_depth_during_last_rebuild =  cmp::max(self.node_depth_during_last_rebuild,depth);

        node_id
    }

    #[inline]
    fn into_iter(self) -> impl Iterator<Item = KP> {
        self.nodes
            .into_iter()
            .map(|node|node.point)
    }
}

pub struct KdTreeNode<F: Float, T> {
    left_node: Option<usize>,
    right_node: Option<usize>,

    point: T,
    dimension: usize,
    split_on: F
}

impl<F: Float, T: KdTreePoint<F>> KdTreeNode<F, T> {
    fn new(p: T, splitting_dimension: usize, split_on_value: F) -> KdTreeNode<F, T> {
        KdTreeNode {
            left_node: None,
            right_node: None,

            point: p,
            dimension: splitting_dimension,
            split_on: split_on_value
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kdtree::test_common::Point2WithId;

    use super::*;

    #[test]
    #[should_panic(expected = "empty vector point not allowed")]
    fn should_panic_given_empty_vector() {
        let mut empty_vec: Vec<Point2WithId> = vec![];

        KdTree::new(&mut empty_vec);
    }

    quickcheck! {
        fn tree_build_creates_tree_with_as_many_leafs_as_there_is_points(xs : Vec<f64>) -> bool {
            if xs.len() == 0 {
                return true;
            }
            let mut vec : Vec<Point2WithId> = vec![];
            for i in 0 .. xs.len() {
                let p = Point2WithId::new(i as i32, xs[i], xs[i]);

                vec.push(p);
            }

            let tree = KdTree::new(&mut qc_value_vec_to_2d_points_vec(&xs));

            let mut to_iterate : Vec<usize> = vec![];
            to_iterate.push(0);

            while to_iterate.len() > 0 {
                let last_index = to_iterate.last().unwrap().clone();
                let ref x = tree.nodes.get(last_index).unwrap();
                to_iterate.pop();
                if x.left_node.is_some() {
                    to_iterate.push(x.left_node.unwrap());
                }
                if x.right_node.is_some() {
                    to_iterate.push(x.right_node.unwrap());
                }
            }
            xs.len() == tree.nodes.len()
        }
    }

    quickcheck! {
        fn nearest_neighbor_search_using_qc(xs : Vec<f64>) -> bool {
            if xs.len() == 0 {
                return true;
            }

            let point_vec = qc_value_vec_to_2d_points_vec(&xs);
            let tree = KdTree::new(&mut point_vec.clone());

            for p in &point_vec {
                let found_nn = tree.nearest_search(p).1;

                assert_eq!(p.id, found_nn.id);
            }

            true
        }
    }

    #[test]
    fn has_neighbor_in_range() {
        let mut vec: Vec<Point2WithId> = vec![Point2WithId::new(0,2.,0.)];

        let tree = KdTree::new(&mut vec);

        assert_eq!(false,tree.has_neighbor_in_range(&Point2WithId::new(0,0.,0.), 0.));
        assert_eq!(false,tree.has_neighbor_in_range(&Point2WithId::new(0,0.,0.), 1.));
        assert_eq!(true,tree.has_neighbor_in_range(&Point2WithId::new(0,0.,0.), 2.));
        assert_eq!(true,tree.has_neighbor_in_range(&Point2WithId::new(0,0.,0.), 300.));
    }

    #[test]
    fn incremental_add_adds_as_expected() {
        //this test is tricky because it can have problems with the automatic tree rebuild.

        let mut vec = vec![Point2WithId::new(0,0.,0.)];

        let mut tree = KdTree::new(&mut vec);

        tree.insert_node(Point2WithId::new(0,1.,0.));
        tree.insert_node(Point2WithId::new(0,-1.,0.));

        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.nodes[0].dimension, 0);

        assert_eq!(tree.nodes[0].left_node.is_some(), true);
        assert_eq!(tree.nodes[1].point.dim(0), 1.);
        assert_eq!(tree.nodes[2].point.dim(0), -1.);

        assert_eq!(tree.nodes[0].right_node.is_some(), true);
    }

    #[test]
    fn incremental_add_filters_duplicates() {
        let mut vec = vec![Point2WithId::new(0,0.,0.)];

        let mut tree = KdTree::new(&mut vec);

        let node = Point2WithId::new(0,1.,0.);
        tree.insert_node(node);
        tree.insert_node(node);

        assert_eq!(tree.nodes.len(), 2);
    }

    fn qc_value_vec_to_2d_points_vec(xs: &Vec<f64>) -> Vec<Point2WithId> {
        let mut vec: Vec<Point2WithId> = vec![];
        for i in 0..xs.len() {
            let mut is_duplicated_value = false;
            for j in 0..i {
                if xs[i] == xs[j] {
                    is_duplicated_value = true;
                    break;
                }
            }
            if !is_duplicated_value {
                let p = Point2WithId::new(i as i32, xs[i], xs[i]);
                vec.push(p);
            }
        }

        vec
    }
}