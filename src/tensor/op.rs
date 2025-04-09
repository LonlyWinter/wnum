use std::fmt::Debug;

use crate::array::{arr::WArr, data::Data, error::WResult};

use super::base::{Grad, WTensor};





#[derive(Clone, Debug)]
pub enum Op<T: Data> {
    Matmul(WTensor<T>, WTensor<T>),
    Add(WTensor<T>, WTensor<T>),
    Sub(WTensor<T>, WTensor<T>),
    Mul(WTensor<T>, WTensor<T>),
    Div(WTensor<T>, WTensor<T>),
    Abs(WTensor<T>),
    Relu(WTensor<T>),
    Exp(WTensor<T>),
    Ln(WTensor<T>),
    Sum(WTensor<T>, u8),
    Max(WTensor<T>, u8),
    Broadcast(WTensor<T>, u8, usize),
    Transpose(WTensor<T>, u8, u8),
    TransposeLast(WTensor<T>),
    Squeeze(WTensor<T>, u8),
    UnSqueeze(WTensor<T>, u8),
    Mean(WTensor<T>, u8),
}



impl<T: Data> WTensor<T> {
    pub fn matmul(&self, rhs: &Self) -> WResult<Self> {
        let op = Op::Matmul(self.clone(), rhs.clone());
        Self::from_op(op)
    }

    pub fn abs(&self) -> WResult<Self> {
        let op = Op::Abs(self.clone());
        Self::from_op(op)
    }

    pub fn relu(&self) -> WResult<Self> {
        let op = Op::Relu(self.clone());
        Self::from_op(op)
    }
    
    pub fn map_item<F: Fn(&T::Basic)->T::Basic>(&self, f: F) -> WResult<Self> {
        let data_res = self.read_data()?.map_item(f)?;
        Self::from_data(data_res, false)
    }

    pub fn sum(&self, dim: u8) -> WResult<Self> {
        let op = Op::Sum(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn sum_all(&self) -> WResult<Self> {
        let d = self.dims()?.dims_num();
        let res = (0..d).fold(self.clone(), | a, b | {
            a.sum(b).unwrap()
        });
        Ok(res)
    }

    pub fn sum_keepdim(&self, dim: u8) -> WResult<Self> {
        let n = self.dims()?.to_vec()[dim as usize];
        self.sum(dim)?.broadcast(dim, n)
    }

    pub fn max(&self, dim: u8) -> WResult<Self> {
        let op = Op::Max(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn max_keepdim(&self, dim: u8) -> WResult<Self> {
        let n = self.dims()?.to_vec()[dim as usize];
        self.max(dim)?.broadcast(dim, n)
    }

    pub fn exp(&self) -> WResult<Self> {
        let op = Op::Exp(self.clone());
        Self::from_op(op)
    }

    pub fn ln(&self) -> WResult<Self> {
        let op = Op::Ln(self.clone());
        Self::from_op(op)
    }

    pub fn softmax(&self, dim: u8) -> WResult<Self> {
        let max = self.max_keepdim(dim)?;
        let diff = self.sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?;
        num.div(&den)
    }

    pub fn log_softmax(&self, dim: u8) -> WResult<Self> {
        let max = self.max_keepdim(dim)?;
        let diff = self.sub(&max)?;
        let num = diff.exp()?;
        let den = num.sum_keepdim(dim)?.ln()?;
        num.div(&den)
    }

    pub fn sqr(&self) -> WResult<Self> {
        self.mul(self)
    }

    pub fn broadcast(&self, dim: u8, n: usize) -> WResult<Self> {
        let op = Op::Broadcast(self.clone(), dim, n);
        Self::from_op(op)
    }

    pub fn t(&self) -> WResult<Self> {
        let op = Op::TransposeLast(self.clone());
        Self::from_op(op)
    }

    pub fn transpose(&self, dim0: u8, dim1: u8) -> WResult<Self> {
        let op = Op::Transpose(self.clone(), dim0, dim1);
        Self::from_op(op)
    }

    pub fn unsqueeze(&self, dim: u8) -> WResult<Self> {
        let op = Op::UnSqueeze(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn squeeze(&self, dim: u8) -> WResult<Self> {
        let op = Op::Squeeze(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn mean(&self, dim: u8) -> WResult<Self> {
        let op = Op::Mean(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn mean_all(&self) -> WResult<Self> {
        let d = self.dims()?.dims_num();
        let res = (0..d).fold(self.clone(), | a, b | {
            a.mean(b).unwrap()
        });
        Ok(res)
    }
}



macro_rules! wtensor_basic_ops {
    ($ty:ident, $method:ident, $method_broadcast:ident) => {
        impl<T: Data> WTensor<T> {
            pub fn $method(&self, rhs: &Self) -> WResult<Self> {
                let op = Op::$ty(self.clone(), rhs.clone());
                Self::from_op(op)
            }

            pub fn $method_broadcast(&self, n: &T::Basic) -> WResult<Self> {
                let data = self.like(n)?;
                self.$method(&data)
            }
        }
    };
}

wtensor_basic_ops!(Add, add, broadcast_add);
wtensor_basic_ops!(Sub, sub, broadcast_sub);
wtensor_basic_ops!(Mul, mul, broadcast_mul);
wtensor_basic_ops!(Div, div, broadcast_div);




impl<T: Data> Op<T> {
    pub fn op_name(&self) -> &str {
        match self {
            Self::Matmul(_, _) => "matmul",
            Self::Add(_, _) => "add",
            Self::Sub(_, _) => "sub",
            Self::Mul(_, _) => "mul",
            Self::Div(_, _) => "div",
            Self::Abs(_) => "abs",
            Self::Relu(_) => "relu",
            Self::Exp(_) => "exp",
            Self::Ln(_) => "ln",
            Self::Sum(_, _) => "sum",
            Self::Max(_, _) => "max",
            Self::Broadcast(_, _, _) => "broadcast",
            Self::Transpose(_, _, _) => "transpose",
            Self::TransposeLast(_) => "transpose_last",
            Self::Squeeze(_, _) => "squeeze",
            Self::UnSqueeze(_, _) => "unsqueeze",
            Self::Mean(_, _) => "mean",
        }
    }

    pub fn get_nodes(&self) -> Vec<WTensor<T>> {
        match self {
            Self::Matmul(a, b) => vec![a.clone(), b.clone()],
            Self::Add(a, b) => vec![a.clone(), b.clone()],
            Self::Sub(a, b) => vec![a.clone(), b.clone()],
            Self::Mul(a, b) => vec![a.clone(), b.clone()],
            Self::Div(a, b) => vec![a.clone(), b.clone()],
            Self::Abs(a) => vec![a.clone(), ],
            Self::Relu(a) => vec![a.clone(), ],
            Self::Exp(a) => vec![a.clone(), ],
            Self::Ln(a) => vec![a.clone(), ],
            Self::Sum(a, _) => vec![a.clone(), ],
            Self::Max(a, _) => vec![a.clone(), ],
            Self::Broadcast(a, _, _) => vec![a.clone(), ],
            Self::Transpose(a, _, _) => vec![a.clone(), ],
            Self::TransposeLast(a) => vec![a.clone(), ],
            Self::Squeeze(a, _) => vec![a.clone(), ],
            Self::UnSqueeze(a, _) => vec![a.clone(), ],
            Self::Mean(a, _) => vec![a.clone(), ],
        }
    }

    pub fn forward(&self) -> WResult<WArr<T>> {
        let res = match self {
            Self::Matmul(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data1data.dims;
                    let data2dim = &data2data.dims;
                    log::debug!("matmul forward {} {} {:?} {:?}", data1.id(), data2.id(), data1dim, data2dim);
                }
                data1data.matmul(&data2data)?
            },
            Self::Add(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data1data.dims;
                    let data2dim = &data2data.dims;
                    log::debug!("add forward {} {} {:?} {:?}", data1.id(), data2.id(), data1dim, data2dim);
                }
                data1data.add(&data2data)?
            },
            Self::Sub(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data1data.dims;
                    let data2dim = &data2data.dims;
                    log::debug!("sub forward {} {} {:?} {:?}", data1.id(), data2.id(), data1dim, data2dim);
                }
                data1data.sub(&data2data)?
            },
            Self::Mul(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data1data.dims;
                    let data2dim = &data2data.dims;
                    log::debug!("mul forward {} {} {:?} {:?}", data1.id(), data2.id(), data1dim, data2dim);
                }
                data1data.mul(&data2data)?
            },
            Self::Div(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data1data.dims;
                    let data2dim = &data2data.dims;
                    log::debug!("mul forward {} {} {:?} {:?}", data1.id(), data2.id(), data1dim, data2dim);
                }
                data1data.div(&data2data)?
            },
            Self::Abs(data1) => {
                data1.read_data()?.abs()?
            },
            Self::Relu(data1) => {
                data1.read_data()?.relu()?
            },
            Self::Exp(data1) => {
                data1.read_data()?.exp()?
            },
            Self::Ln(data1) => {
                data1.read_data()?.ln()?
            },
            Self::Sum(data1, dim) => {
                let data = data1.read_data()?;
                let data_res = data.sum(*dim)?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data.dims;
                    log::debug!("sum forward {} {:?} {:?}", data1.id(), data1dim, data_res);
                }
                data_res
            },
            Self::Max(data1, dim) => {
                let data = data1.read_data()?;
                let data_res = data.max(*dim)?;
                #[cfg(feature = "logger")]
                {
                    let data1dim = &data.dims;
                    log::debug!("max forward {} {:?} {:?}", data1.id(), data1dim, data_res);
                }
                data_res
            },
            Self::Broadcast(data1, dim, n) => {
                data1.read_data()?.broadcast(*dim, *n)?
            },
            Self::TransposeLast(data1) => {
                data1.read_data()?.t()?
            },
            Self::Transpose(data1, dim0, dim1) => {
                data1.read_data()?.transpose(*dim0, *dim1)?
            },
            Self::UnSqueeze(data1, dim) => {
                data1.read_data()?.unsqueeze(*dim)?
            },
            Self::Squeeze(data1, dim) => {
                data1.read_data()?.squeeze(*dim)?
            },
            Self::Mean(data1, dim) => {
                data1.read_data()?.mean(*dim)?
            }
        };
        Ok(res)
    }

    pub fn backward(&self, grads: &mut Grad<T>, node_id: u16, node_data: &WArr<T>) -> WResult<()> {
        #[cfg(feature = "logger")]
        log::debug!("backward {} {} {:?}", self.op_name(), node_id, node_data);
        let node_grad = grads.get(node_id).expect("node grad not found").clone();
        match self {
            Self::Matmul(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward: node_grad {:?}, data1: {:?}, data2: {:?}", node_grad, data1data, data2data);
                let data1t = data1data.t()?;
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward data1t {:?}", data1t);
                let data2t = data2data.t()?;
                // {
                //     log::debug!("save data2data");
                //     let mut f = File::create("temp1.json")?;
                //     let data = data2data
                //         .to_vec()?
                //         .into_iter()
                //         .map(| v | format!("{}", v))
                //         .collect::<Vec<_>>();
                //     let data = serde_json::to_string(&data)?;
                //     f.write_all(data.as_bytes())?;
                    
                //     log::debug!("save data2t");
                //     let mut f = File::create("temp2.json")?;
                //     let data = data2t
                //         .to_vec()?
                //         .into_iter()
                //         .map(| v | format!("{}", v))
                //         .collect::<Vec<_>>();
                //     let data = serde_json::to_string(&data)?;
                //     f.write_all(data.as_bytes())?;
                // }
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward data2t {:?} {:?}", data2data, data2t);
                let grad1 = node_grad.matmul(&data2t)?;
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward grad1 {:?}", grad1);
                let grad2 = data1t.matmul(&node_grad)?;
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward grad2 {:?}", grad2);
                // #[cfg(feature = "logger")]
                // log::debug!("matmul backward {} {} {:?} {:?}", data1.id(), data2.id(), grad1.dims, grad2.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
                grads.insert(
                    data2.id(),
                    grad2
                )?;
            },
            Self::Add(data1, data2) => {
                #[cfg(feature = "logger")]
                log::debug!("add backward {} {} {:?}", data1.id(), data2.id(), node_grad.dims);
                grads.insert(
                    data1.id(),
                    node_grad.clone()
                )?;
                grads.insert(
                    data2.id(),
                    node_grad
                )?;
            },
            Self::Sub(data1, data2) => {
                #[cfg(feature = "logger")]
                log::debug!("sub backward {} {} {:?}", data1.id(), data2.id(), node_grad.dims);
                grads.insert(
                    data1.id(),
                    node_grad.clone()
                )?;
                grads.insert(
                    data2.id(),
                    node_grad.neg()?
                )?;
            },
            Self::Mul(data1, data2) => {
                // #[cfg(feature = "logger")]
                // log::debug!("mul backward {} {}", node_grad, data1.read_data()?);
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                let grad1 = node_grad.mul(&data2data)?;
                let grad2 = node_grad.mul(&data1data)?;
                #[cfg(feature = "logger")]
                log::debug!("mul backward {} {} {:?} {:?}", data1.id(), data2.id(), grad1.dims, grad2.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
                grads.insert(
                    data2.id(),
                    grad2
                )?;
            },
            Self::Div(data1, data2) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                let grad1 = node_grad.div(&data2data)?;
                let t = data2data.mul(&data2data)?;
                let grad2 = node_grad.mul(&data1data)?.div(&t)?;
                #[cfg(feature = "logger")]
                log::debug!("mul backward {} {} {:?} {:?}", data1.id(), data2.id(), grad1.dims, grad2.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
                grads.insert(
                    data2.id(),
                    grad2
                )?;
            },
            Self::Abs(data1) => {
                let t0 = data1.read_data()?;
                let t1 = t0.where_cond(
                    &t0.zero(),
                    &t0.one(),
                    &t0.neg_one(),
                )?;
                let grad1 = node_grad.mul(&t1)?;
                #[cfg(feature = "logger")]
                log::debug!("abs backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Relu(data1) => {
                let t0 = data1.read_data()?;
                let t1 = t0.where_cond(
                    &t0.zero(),
                    &t0.one(),
                    &t0.zero(),
                )?;
                let grad1 = node_grad.mul(&t1)?;
                #[cfg(feature = "logger")]
                log::debug!("relu backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Exp(data1) => {
                let grad1 = node_grad.mul(node_data)?;
                #[cfg(feature = "logger")]
                log::debug!("exp backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Ln(data1) => {
                let data1data = data1.read_data()?;
                let grad1 = node_grad.div(&data1data)?;
                #[cfg(feature = "logger")]
                log::debug!("ln backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Sum(data1, dim) => {
                let n = data1.dims()?.to_vec()[*dim as usize];
                let grad1 = node_grad.broadcast(*dim, n)?;
                #[cfg(feature = "logger")]
                log::debug!("sum backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Max(data1, dim) => {
                #[cfg(feature = "logger")]
                log::debug!("max backward start");
                let n = data1.dims()?.to_vec()[*dim as usize];
                let data1data = data1.read_data()?;
                let node_data_now = node_data.broadcast(*dim, n)?;
                let v = data1data.eq_item(&node_data_now)?;
                let grad1 = node_grad.broadcast(*dim, n)?.mul(&v)?;
                #[cfg(feature = "logger")]
                log::debug!("max backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Broadcast(data1, dim, _n) => {
                let grad1 = node_grad.sum(*dim)?;
                #[cfg(feature = "logger")]
                log::debug!("broadcast backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::TransposeLast(data1) => {
                let grad1 = node_grad.t()?;
                #[cfg(feature = "logger")]
                log::debug!("TransposeLast backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Transpose(data1, dim0, dim1) => {
                let grad1 = node_grad.transpose(*dim0, *dim1)?;
                #[cfg(feature = "logger")]
                log::debug!("Transpose backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::UnSqueeze(data1, dim) => {
                #[cfg(feature = "logger")]
                log::debug!("UnSqueeze backward start");
                let grad1 = node_grad.squeeze(*dim)?;
                #[cfg(feature = "logger")]
                log::debug!("UnSqueeze backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Squeeze(data1, dim) => {
                #[cfg(feature = "logger")]
                log::debug!("Squeeze backward start");
                let grad1 = node_grad.squeeze(*dim)?;
                #[cfg(feature = "logger")]
                log::debug!("Squeeze backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Mean(data1, dim) => {
                #[cfg(feature = "logger")]
                log::debug!("mean backward start");
                let n0 = data1.dims()?.to_vec()[*dim as usize];
                let n1 = T::usize_to_basic(n0)?;
                let grad1 = node_grad.broadcast_div(&n1)?;
                let grad1 = grad1.broadcast(*dim, n0)?;
                #[cfg(feature = "logger")]
                log::debug!("mean backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            }
        }
        Ok(())
    }
}


// macro_rules! wtensor_float_op {
//     ($ty:ident, $method:ident) => {
//         impl WTensor<$ty> {
//             pub fn sum_exp(&self, dim: u8, broadcast: bool) -> WResult<Self> {
//                 let data = self.read_data()?;
//                 let n = data.dims.to_vec()[dim as usize];
//                 let data_res = data.sum_exp(dim)?;
//                 log::debug!("softmax res: {}", data_res);
//                 let data_res = if broadcast {
//                     data_res.broadcast(dim, n)?
//                 } else {
//                     data_res
//                 };
//                 Ok(Self::from_data(data_res, false))
//             }

//             pub fn exp(&self) -> WResult<Self> {
//                 let op = Op::$method(self.clone());
//                 Self::from_op(op)
//             }
//         }
//     }
// }

// wtensor_float_op!(f32, Exp1);
// wtensor_float_op!(f64, Exp2);

