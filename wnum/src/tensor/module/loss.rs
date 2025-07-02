
use crate::{array::{data::Data, error::{WError, WResult}}, tensor::base::WTensor};


pub fn mse<T: Data>(a: &WTensor<T>, b: &WTensor<T>) -> WResult<WTensor<T>> {
    a.sub(b)?.sqr()?.mean_all()
}

pub fn nll<T: Data>(a: &WTensor<T>, b: &WTensor<T>) -> WResult<WTensor<T>> {
    let b_dim = b.dims()?;
    if b_dim.dims_num() != 1 {
        return Err(WError::DimNumError("B Need 1 dim".to_string()));
    }
    let b_n = b_dim.dims_len();
    if !a.dim(0)?.eq(&b_n) {
        return Err(WError::DimNumError("A dim0".to_string()));
    }
    let b_n = T::f64_to_basic(-1f64 / b_n as f64)?;
    a.gather(&b.unsqueeze(1)?, 1)?.sum_all()?.broadcast_mul(&b_n)
}

pub fn cross_entropy<T: Data>(a: &WTensor<T>, b: &WTensor<T>) -> WResult<WTensor<T>> {
    if a.dims()?.dims_num() != 2 {
        return Err(WError::DimNumError("A Need 2 dims".to_string()));
    }
    #[cfg(feature = "logger")]
    log::debug!("log_softmax a {:?}, dim {:?}", a, 1);
    let a = a.log_softmax(1)?;
    #[cfg(feature = "logger")]
    log::debug!("cross_entropy a {:?}, b {:?}", a, b);
    nll(&a, b)
}