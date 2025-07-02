use crate::{array::{data::Data, error::WResult}, tensor::{base::WTensor, varmap::Varmap}};




pub struct Norm<T: Data> {
    weight: WTensor<T>,
    bias: Option<WTensor<T>>,
    remove_mean: bool,
    eps: T::Basic,
}

impl<T: Data> Norm<T> {
    pub fn new_with_data(weight: WTensor<T>, bias: Option<WTensor<T>>, eps: f64, remove_mean: bool) -> WResult<Self> {
        let eps = T::f64_to_basic(eps)?;
        Ok(Self {
            weight,
            bias,
            remove_mean,
            eps
        })
    }

    pub fn new_no_varmap(size: usize, eps: f64, remove_mean: bool, bias: bool) -> WResult<Self> {
        let weight = WTensor::ones(size)?;
        let bias = if bias {
            Some(WTensor::zeros(size)?)
        } else {
            None
        };
        let eps = T::f64_to_basic(eps)?;
        Ok(Self {
            weight,
            bias,
            remove_mean,
            eps
        })
    }

    pub fn new_with_config(size: usize, eps: f64, remove_mean: bool, bias: bool, name: &str, varmap: &mut Varmap<T>) -> WResult<Self> {
        let data = Self::new_no_varmap(size, eps, remove_mean, bias)?;
        varmap.add_tensor(&format!("{name}.w"), &data.weight);
        if bias {
            varmap.add_tensor(&format!("{name}.b"), data.bias.as_ref().unwrap());
        }
        Ok(data)
    }

    pub fn new_layer_nobias(size: usize, eps: f64, name: &str, varmap: &mut Varmap<T>) -> WResult<Self> {
        let data = Self::new_no_varmap(size, eps, true, false)?;
        varmap.add_tensor(&format!("{name}.w"), &data.weight);
        Ok(data)
    }

    pub fn new_layer(size: usize, eps: f64, name: &str, varmap: &mut Varmap<T>) -> WResult<Self> {
        let data = Self::new_no_varmap(size, eps, true, true)?;
        varmap.add_tensor(&format!("{name}.w"), &data.weight);
        varmap.add_tensor(&format!("{name}.b"), data.bias.as_ref().unwrap());
        Ok(data)
    }

    pub fn new_rms(size: usize, eps: f64, name: &str, varmap: &mut Varmap<T>) -> WResult<Self> {
        let data = Self::new_no_varmap(size, eps, false, false)?;
        varmap.add_tensor(&format!("{name}.w"), &data.weight);
        Ok(data)
    }

    pub fn forward(&self, x: &WTensor<T>) -> WResult<WTensor<T>> {
        todo!()
    }
}