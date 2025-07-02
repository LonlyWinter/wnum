
#[cfg(feature = "random")]
use rand::{distr::{Distribution, StandardUniform}, rng};
#[cfg(feature = "random")]
use rand_distr::Normal;
#[cfg(feature = "random")]
use rand_distr::Uniform;

use super::{arr::WArr, data::Data, dim::WArrDims, error::WResult};




#[cfg(feature = "random")]
impl<T: Data> WArr<T> {
    pub fn random_uniform<F: Into<WArrDims>>(dims_raw: F) -> WResult<Self> {
        let dims = dims_raw.into();
        let dims_all = dims.dims_len();
        let m = 1.0 / (dims_all as f64).sqrt();
        let data = Uniform::new(-m, m).unwrap()
            .sample_iter(rng())
            .take(dims_all)
            .map(| v | T::f64_to_basic(v))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    pub fn random_normal<F: Into<WArrDims>>(dims_raw: F) -> WResult<Self> {
        let dims: WArrDims = dims_raw.into();
        let mut dims_all = dims.to_vec();
        let dims_first = dims_all.drain(0..2).collect::<Vec<_>>();
        let dims_a: usize = dims_first.iter().sum::<usize>() / 2;
        // let dims_a = dims_first[1];
        let dims_b: usize = dims_all.iter().product();
        let dims_c = dims.dims_len();
        let m = 2.0f64.sqrt() / ((dims_a * dims_b) as f64).sqrt();
        let data = Normal::new(0.0, m).unwrap()
            .sample_iter(rng())
            .take(dims_c)
            .map(| v | T::f64_to_basic(v))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    pub fn like_random(&self) -> WResult<Self> {
        let dims = self.dims.clone();
        Self::random_uniform(dims)
    }
}



impl<T: Data> WArr<T> {
    pub fn ones<F: Into<WArrDims>>(dims_raw: F) -> WResult<Self> {
        let dims = dims_raw.into();
        let dims_all = dims.dims_len();
        let data = vec![1.0f64; dims_all]
            .into_iter()
            .map(| v | T::f64_to_basic(v))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    pub fn zeros<F: Into<WArrDims>>(dims_raw: F) -> WResult<Self> {
        let dims = dims_raw.into();
        let dims_all = dims.dims_len();
        let data = vec![0.0f64; dims_all]
            .into_iter()
            .map(| v | T::f64_to_basic(v))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    pub fn ones_like(&self) -> WResult<Self> {
        Self::ones(self.dims.clone())
    }

    pub fn zeros_like(&self) -> WResult<Self> {
        Self::zeros(self.dims.clone())
    }

    pub fn from_shape<F: Into<WArrDims>, V: Into<f64>>(dims_raw: F, n: V) -> WResult<Self> {
        let dims = dims_raw.into();
        let dims_all = dims.dims_len();
        let data = vec![n.into(); dims_all]
            .into_iter()
            .map(| v | T::f64_to_basic(v))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(Self::new(data, dims))
    }
}




#[cfg(feature = "random")]
impl<T, F> From<F> for WArr<T>
where
    T: Data,
    F: Into<WArrDims>,
    StandardUniform: Distribution<T>,
{
    fn from(value: F) -> Self {
        Self::zeros(value).unwrap()
    }
}  
