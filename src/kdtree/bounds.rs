use num_traits::Float;
use crate::kdtree::KdTreePoint;

#[derive(Clone, Copy)]
pub struct Bounds<F: Float> {
    pub bounds: [(F, F); 3],

    widest_dim: usize,
    midvalue_of_widest_dim: F,
}

impl<F: Float> Bounds<F> {
    pub fn new_from_points<T: KdTreePoint<F>>(points: &[T]) -> Bounds<F> {
        let mut bounds = Bounds {
            bounds: [(F::zero(), F::zero()), (F::zero(), F::zero()), (F::zero(), F::zero())],
            widest_dim: 0,
            midvalue_of_widest_dim: F::zero(),
        };

        for i in 0..points[0].dims() {
            bounds.bounds[i].0 = points[0].dim(i);
            bounds.bounds[i].1 = points[0].dim(i);
        }

        for v in points.iter() {
            for dim in 0..v.dims() {
                bounds.bounds[dim].0 = bounds.bounds[dim].0.min(v.dim(dim));
                bounds.bounds[dim].1 = bounds.bounds[dim].1.max(v.dim(dim));
            }
        }

        bounds.calculate_variables();

        bounds
    }

    #[inline]
    pub fn get_widest_dim(&self) -> usize {
        self.widest_dim
    }

    #[inline]
    pub fn get_midvalue_of_widest_dim(&self) -> F {
        self.midvalue_of_widest_dim
    }

    #[inline]
    pub fn clone_moving_max(&self, value: F, dimension: usize) -> Bounds<F> {
        let mut cloned = Bounds {
            bounds: self.bounds.clone(),
            ..*self
        };

        cloned.bounds[dimension].1 = value;

        cloned.calculate_variables();

        cloned
    }

    pub fn clone_moving_min(&self, value: F, dimension: usize) -> Bounds<F> {
        let mut cloned = Bounds {
            bounds: self.bounds.clone(),
            ..*self
        };
        cloned.bounds[dimension].0 = value;

        cloned.calculate_variables();

        cloned
    }

    fn calculate_widest_dim(&mut self) {
        let mut widest_dimension = 0usize;
        let mut max_found_spread = self.bounds[0].1 - self.bounds[0].0;

        for i in 0..self.bounds.len() {
            let dimension_spread = self.bounds[i].1 - self.bounds[i].0;

            if dimension_spread > max_found_spread {
                max_found_spread = dimension_spread;
                widest_dimension = i;
            }
        }

        self.widest_dim = widest_dimension;
    }

    fn calculate_variables(&mut self) {
        self.calculate_widest_dim();
        self.midvalue_of_widest_dim = (self.bounds[self.get_widest_dim()].0 + self.bounds[self.get_widest_dim()].1) / F::from(2.0f32).unwrap();
    }
}


#[cfg(test)]
mod tests {
    use super::Bounds;
    use crate::kdtree::test_common::Point2WithId;

    #[test]
    fn bounds_test() {
        let p1 = Point2WithId::new(1, 1.0, 0.5);
        let p2 = Point2WithId::new(1, 3.0, 4.0);
        let v = vec![p1, p2];


        let bounds = Bounds::new_from_points(&v);

        assert_eq!((1., 3.0), bounds.bounds[0]);
        assert_eq!((0.5, 4.0), bounds.bounds[1]);

        assert_eq!(1, bounds.get_widest_dim());
    }
}
