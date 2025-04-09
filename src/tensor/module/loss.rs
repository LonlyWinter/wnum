
use crate::{array::{data::Data, error::WResult}, tensor::base::WTensor};


pub fn mse<T: Data>(a: &WTensor<T>, b: &WTensor<T>) -> WResult<WTensor<T>> {
    a.sub(b)?.sqr()?.mean_all()
}