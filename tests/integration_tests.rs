extern crate kdtree;
extern crate rand;

use rand::Rng;

use kdtree::kdtree::test_common::*;
use kdtree::kdtree::KdTreePoint;
use kdtree::kdtree::distance::squared_euclidean;

fn gen_random() -> f64 {
    rand::thread_rng().gen_range(0., 1000.)
}

fn find_nn_with_linear_search(points : &Vec<Point3WithId>, find_for : Point3WithId) -> (f64, &Point3WithId) {
    let mut best_found_distance =  squared_euclidean(find_for.dims(), points[0].dims());
    let mut closed_found_point = &points[0];

    for p in points {
        let dist = squared_euclidean(find_for.dims(), p.dims());

        if dist < best_found_distance {
            best_found_distance = dist;
            closed_found_point = &p;
        }
    }

    (best_found_distance, closed_found_point)
}

fn find_neigbours_with_linear_search(points : &Vec<Point3WithId>, find_for : Point3WithId, dist: f64) -> Vec<(f64, &Point3WithId)> {
    let mut result = Vec::new();

    for p in points {
        let d = squared_euclidean(find_for.dims(), p.dims());

        if d <= dist {
            result.push((d, p));
        }
    }

    result
}

fn generate_points(point_count : usize) -> Vec<Point3WithId> {
    let mut points : Vec<Point3WithId> = vec![];

    for i in 0 .. point_count {
        points.push(Point3WithId::new(i as i32, gen_random(),gen_random(),gen_random()));
    }

    points
}


#[test]
fn test_against_1000_random_points() {
    let point_count = 1000usize;
    let points = generate_points(point_count);
    kdtree::kdtree::test_common::Point1WithId::new(0,0.);

    let tree = kdtree::kdtree::KdTree::new(&mut points.clone());

    //test points pushed into the tree, id should be equal.
    for i in 0 .. point_count {
        let p = &points[i];

        assert_eq!(p.id, tree.nearest_search(p).1.id );
    }

    //test randomly generated points within the cube. and do the linear search. should match
    for _ in 0 .. 500 {
        let p = Point3WithId::new(0i32, gen_random(), gen_random(), gen_random());

        let found_by_linear_search = find_nn_with_linear_search(&points, p);
        let point_found_by_kdtree = tree.nearest_search(&p);

        assert_eq!(point_found_by_kdtree.1.id, found_by_linear_search.1.id);
    }
}

#[test]
fn test_incrementally_build_tree_against_built_at_once() {
    let point_count = 2000usize;
    let mut points = generate_points(point_count);

    let tree_built_at_once = kdtree::kdtree::KdTree::new(&mut points.clone());
    let mut tree_built_incrementally = kdtree::kdtree::KdTree::new(&mut points[0..1]);

    for i in 1 .. point_count {
        let p = &points[i];

        tree_built_incrementally.insert_node(p.clone());
    }


    //test points pushed into the tree, id should be equal.
    for i in 0 .. point_count {
        let p = &points[i];

        assert_eq!(tree_built_at_once.nearest_search(p).1.id, tree_built_incrementally.nearest_search(p).1.id);
    }


    //test randomly generated points within the cube. and do the linear search. should match
    for _ in 0 .. 5000 {
        let p = Point3WithId::new(0i32, gen_random(), gen_random(), gen_random());
        assert_eq!(tree_built_at_once.nearest_search(&p).1.id, tree_built_incrementally.nearest_search(&p).1.id);
    }
}


#[test]
fn test_neighbour_search_with_distance() {
    let point_count = 1000usize;
    let points = generate_points(point_count);
    let tree = kdtree::kdtree::KdTree::new(&mut points.clone());

    for _ in 0 .. 500 {
        let dist = 100.0;
        let p = Point3WithId::new(0i32, gen_random(), gen_random(), gen_random());

        let mut found_by_linear_search = find_neigbours_with_linear_search(&points, p, dist * dist);
        let mut point_found_by_kdtree: Vec<_> = tree.nearest_search_dist(p, dist * dist).collect();

        assert_eq!(found_by_linear_search.len(), point_found_by_kdtree.len());

        if point_found_by_kdtree.len() > 0 {
            found_by_linear_search.sort_by(|a, b| a.1.id.cmp(&b.1.id));
            point_found_by_kdtree.sort_by(|a, b| a.1.id.cmp(&b.1.id));
        }

        assert_eq!(point_found_by_kdtree, found_by_linear_search);
    }
}