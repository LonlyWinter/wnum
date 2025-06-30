
use crate::{array::{arr::WArr, data::Data, error::WResult}, tensor::base::WTensor};


pub trait Optim<T: Data> {
    fn step(&mut self, node: WArr<T>, grad: WArr<T>) -> WResult<WArr<T>>;
    
    fn backward(&mut self, loss: WTensor<T>) -> WResult<()> {
        let nodes: Vec<WTensor<T>> = loss.get_nodes();
        let mut grads = loss.backward()?;
        // 设置结果
        for node in nodes.iter() {
            if !node.trace() {
                continue;
            }
            let grad = grads.remove(node.id()).expect("No grad found");
            let node_data = node.read_data()?.clone();
            #[cfg(feature = "logger")]
            log::debug!("backward, node_data before: {node}, {grad}");
            let data_temp = self.step(node_data, grad)?;
            node.write_data(data_temp)?;
            #[cfg(feature = "logger")]
            log::debug!("backward, node_data after: {node}");
        }
        Ok(())
    }
}

pub struct SGD<T: Data> {
    lr: T::Basic,
}

impl<T: Data> SGD<T> {
    pub fn new(lr: T::Basic) -> Self {
        Self { lr }
    }
}

impl<T: Data> Optim<T> for SGD<T> {
    fn step(&mut self, node: WArr<T>, grad: WArr<T>) -> WResult<WArr<T>> {
        let grad_now = grad.broadcast_mul(&self.lr)?;
        let res = node.sub(&grad_now)?;
        Ok(res)
    }
}