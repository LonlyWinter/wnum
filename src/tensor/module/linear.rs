
use rand::rng;
use rand_distr::{Distribution, Uniform};

use crate::{array::{arr::WArr, data::Data, error::{WError, WResult}}, tensor::{base::WTensor, varmap::Varmap}};



pub struct Linear<T: Data> {
    pub w: WTensor<T>,
    pub b: Option<WTensor<T>>
}

impl<T: Data> Linear<T> {
    pub fn new(dim_inp: usize, dim_opt: usize, bias: bool, name: &str, varmap: &mut Varmap<T>) -> WResult<Self> {
        let b = if bias {
            let m = 1.0 / (dim_inp as f64).sqrt();
            let data = Uniform::new(-m, m).unwrap()
                .sample_iter(rng())
                .take(dim_opt)
                .map(| v | T::f64_to_basic(v))
                .collect::<WResult<Vec<_>>>()?;
            let data = WArr::from_vec2(vec![data])?;
            let b = WTensor::from_data(data, true)?;
            varmap.add_tensor(&format!("{}.b", name), &b);
            Some(b)
        } else {
            None
        };
        let w = varmap.gen_tensor_random_normal(
            &format!("{}.w", name),
            (dim_inp, dim_opt)
        )?;
        Ok(Self { w, b })
    }

    pub fn new_with_tensor(w: WTensor<T>, b: Option<WTensor<T>>) -> WResult<Self> {
        let dim_opt_w = w.dims()?.to_vec().last().cloned().unwrap();
        if let Some(b) = &b {
            let dim_opt_b = b.dims()?.to_vec().last().cloned().unwrap();
            if dim_opt_w != dim_opt_b {
                return Err(WError::DimMisMatch(dim_opt_w, dim_opt_b));
            }
        }
        Ok(Self { w, b })
    }


    pub fn forward(&self, data: &WTensor<T>) -> WResult<WTensor<T>> {
        log::debug!("linear x: {} {}", data, self.w);
        let batch_num = data.dims()?.to_vec()[0];
        let y = data.matmul(&self.w)?;
        log::debug!("linear w*x: {} {}", y.id(), y.read_data()?);
        let y = if let Some(b) = &self.b {
            let b = b.broadcast(0, batch_num)?;
            y.add(&b)?
        } else {
            y
        };
        log::debug!("linear w*x+b: {} {}", y.id(), y.read_data()?);
        Ok(y)
    }
}
