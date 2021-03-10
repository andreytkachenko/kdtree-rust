use super::KdTreePoint;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point3WithId {
    dims: [f64; 3],
    pub id: i32,
}

impl Point3WithId {
    pub fn new(id: i32, x: f64, y: f64, z: f64) -> Point3WithId {
        Point3WithId {
            dims: [x, y, z],
            id: id,
        }
    }
}

impl KdTreePoint<f64> for Point3WithId {
    #[inline]
    fn dims(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    fn dim(&self, i: usize) -> f64 {
        self.dims[i]
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point2WithId {
    dims: [f64; 2],
    pub id: i32,
}

impl Point2WithId {
    pub fn new(id: i32, x: f64, y: f64) -> Point2WithId {
        Point2WithId {
            dims: [x, y],
            id: id,
        }
    }
}

impl KdTreePoint<f64> for Point2WithId {
    #[inline]
    fn dims(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    fn dim(&self, i: usize) -> f64 {
        self.dims[i]
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point1WithId {
    dims: [f64; 1],
    pub id: i32,
}

impl Point1WithId {
    pub fn new(id: i32, x: f64) -> Point1WithId {
        Point1WithId {
            dims: [x],
            id: id,
        }
    }
}

impl KdTreePoint<f64> for Point1WithId {
    #[inline]
    fn dims(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    fn dim(&self, i: usize) -> f64 {
        self.dims[i]
    }
}


#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Features{
    feature_a: f64,
    feature_b: f64,
    feature_c: f64,
    feature_r: f64,
    feature_x: f64,
}

impl Features{
    #[inline]
    pub fn new(a: f64, b: f64, c: f64, r: f64, x: f64) -> Self{
        Self{
            feature_a: a, feature_b: b, feature_c: c, feature_r: r, feature_x: x
        }
    }
}

impl KdTreePoint<f64> for Features {
    #[inline]
    fn dims(&self) -> usize {
        5usize
    }

    #[inline]
    fn dim(&self, i: usize) -> f64 {
        match i{
            0 => self.feature_a,
            1 => self.feature_b,
            2 => self.feature_c,
            3 => self.feature_r,
            4 => self.feature_x,
            _ => unreachable!()
        }
    }
}
