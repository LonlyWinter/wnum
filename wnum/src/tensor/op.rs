use std::fmt::Debug;

use crate::array::{arr::WArr, data::Data, error::{WError, WResult}};

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
    Gather(WTensor<T>, WTensor<T>, u8),
    Sqr(WTensor<T>),
    Sqrt(WTensor<T>),
    Cat(Vec<WTensor<T>>, u8),
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
        let n = self.dim(dim)?;
        self.sum(dim)?.broadcast(dim, n)
    }

    pub fn max(&self, dim: u8) -> WResult<Self> {
        let op = Op::Max(self.clone(), dim);
        Self::from_op(op)
    }

    pub fn max_keepdim(&self, dim: u8) -> WResult<Self> {
        let n = self.dim(dim)?;
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
        diff.sub(&den)
    }

    pub fn sqr(&self) -> WResult<Self> {
        let op = Op::Sqr(self.clone());
        Self::from_op(op)
    }

    pub fn sqrt(&self) -> WResult<Self> {
        let op = Op::Sqrt(self.clone());
        Self::from_op(op)
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

    pub fn gather(&self, indexes: &Self, dim: u8) -> WResult<Self> {
        let op = Op::Gather(self.clone(), indexes.clone(), dim);
        Self::from_op(op)
    }

    pub fn cat(datas: Vec<&Self>, dim: u8) -> WResult<Self> {
        if datas.len() < 2 {
            return Err(WError::DimNumError(format!("Need >= 2 dims")));
        }
        let datas = datas.into_iter().map(| v | v.to_owned()).collect::<Vec<_>>();
        let op = Op::Cat(datas, dim);
        Self::from_op(op)
    }
}



macro_rules! wtensor_basic_ops {
    ($ty:ident, $method:ident, $method_broadcast:ident, $method_broadcast2:ident) => {
        impl<T: Data> WTensor<T> {
            pub fn $method(&self, rhs: &Self) -> WResult<Self> {
                let op = Op::$ty(self.clone(), rhs.clone());
                Self::from_op(op)
            }

            pub fn $method_broadcast(&self, n: &T::Basic) -> WResult<Self> {
                let data = self.like(n)?;
                self.$method(&data)
            }

            pub fn $method_broadcast2(&self, rhs: &Self) -> WResult<Self> {
                let mut dims = self.dims()?.to_vec();
                let mut dims_w = rhs.dims()?.to_vec();
                let mut w = rhs.clone();
                if dims.len() > 0 {
                    dims_w.reverse();
                    for dim in dims_w {
                        let d = dims.pop().unwrap();
                        if d != dim {
                            return Err(WError::DimMisMatch(d, dim));
                        }
                    }
                    dims.reverse();
                    for dim in dims.iter() {
                        w = w.unsqueeze(0)?.broadcast(0, *dim)?;
                    }
                }
                self.$method(&w)
            }
        }
    };
}

wtensor_basic_ops!(Add, add, broadcast_add, add_broadcast);
wtensor_basic_ops!(Sub, sub, broadcast_sub, sub_broadcast);
wtensor_basic_ops!(Mul, mul, broadcast_mul, mul_broadcast);
wtensor_basic_ops!(Div, div, broadcast_div, div_broadcast);




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
            Self::Gather(_, _, _) => "gather",
            Self::Sqr(_) => "sqr",
            Self::Sqrt(_) => "sqrt",
            Self::Cat(_, _) => "cat",
        }
    }

    pub fn op_id(&self) -> u16 {
        match self {
            Self::Matmul(a, b) => a.id().max(b.id()),
            Self::Add(a, b) => a.id().max(b.id()),
            Self::Sub(a, b) => a.id().max(b.id()),
            Self::Mul(a, b) => a.id().max(b.id()),
            Self::Div(a, b) => a.id().max(b.id()),
            Self::Abs(a) => a.id(),
            Self::Relu(a) => a.id(),
            Self::Exp(a) => a.id(),
            Self::Ln(a) => a.id(),
            Self::Sum(a, _) => a.id(),
            Self::Max(a, _) => a.id(),
            Self::Broadcast(a, _, _) => a.id(),
            Self::Transpose(a, _, _) => a.id(),
            Self::TransposeLast(a) => a.id(),
            Self::Squeeze(a, _) => a.id(),
            Self::UnSqueeze(a, _) => a.id(),
            Self::Mean(a, _) => a.id(),
            Self::Gather(a, b, _) => a.id().max(b.id()),
            Self::Sqr(a) => a.id(),
            Self::Sqrt(a) => a.id(),
            Self::Cat(a, _) => a.iter().map(| v | v.id()).max().unwrap(),
        }
    }

    pub fn get_nodes(&self) -> Vec<WTensor<T>> {
        match self {
            Self::Matmul(a, b) => vec![b.clone(), a.clone()],
            Self::Add(a, b) => vec![b.clone(), a.clone()],
            Self::Sub(a, b) => vec![b.clone(), a.clone()],
            Self::Mul(a, b) => vec![b.clone(), a.clone()],
            Self::Div(a, b) => vec![b.clone(), a.clone()],
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
            Self::Gather(a, b, _) => vec![b.clone(), a.clone()],
            Self::Sqr(a) => vec![a.clone(), ],
            Self::Sqrt(a) => vec![a.clone(), ],
            Self::Cat(a, _) => a.to_owned(),
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
            },
            Self::Gather(data1, data2, dim) => {
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                data1data.gather(&data2data, *dim)?
            },
            Self::Sqr(data1) => {
                let data = data1.read_data()?;
                data.mul(&data)?
            },
            Self::Sqrt(data1) => {
                data1.read_data()?.sqrt()?
            },
            Self::Cat(datas, dim) => {
                let data0 = datas[0].read_data()?;
                let data1 = datas[1].read_data()?;
                let data = data0.concat(&data1, *dim)?;
                datas.iter()
                    .skip(2)
                    .map(| v | v.read_data())
                    .collect::<WResult<Vec<_>>>()?
                    .into_iter()
                    .fold(data, | a, b | {
                        a.concat(&b, *dim).unwrap()
                    })
            },
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
                    data2.id(),
                    node_grad.neg()?
                )?;
                grads.insert(
                    data1.id(),
                    node_grad
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
                let t = data2data.sqr()?;
                let grad2 = node_grad.mul(&data1data)?.div(&t)?.neg()?;
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
                let n = data1.dim(*dim)?;
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
                let n = data1.dim(*dim)?;
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
                let n0 = data1.dim(*dim)?;
                let n1 = T::usize_to_basic(n0)?;
                let grad1 = node_grad.broadcast_div(&n1)?;
                let grad1 = grad1.broadcast(*dim, n0)?;
                #[cfg(feature = "logger")]
                log::debug!("mean backward {} {:?}", data1.id(), grad1.dims);
                grads.insert(
                    data1.id(),
                    grad1
                )?;
            },
            Self::Gather(data1, data2, dim) => {
                #[cfg(feature = "logger")]
                log::debug!("gather backward start");
                let data1data = data1.read_data()?;
                let data2data = data2.read_data()?;
                let grad = data1data.ones_like()?
                    .scatter(&data2data, &node_grad, *dim)?;
                grads.insert(
                    data1.id(),
                    grad
                )?;
            },
            Self::Sqr(data1) => {
                #[cfg(feature = "logger")]
                log::debug!("sqr backward start");
                let grad = data1.read_data()?
                    .mul(&node_grad)?
                    .broadcast_mul(&T::f64_to_basic(2.0)?)?;
                grads.insert(
                    data1.id(),
                    grad
                )?;
            },
            Self::Sqrt(data1) => {
                #[cfg(feature = "logger")]
                log::debug!("sqrt backward start");
                let grad = node_grad.div(node_data)?
                    .broadcast_mul(&T::f64_to_basic(0.5)?)?;
                grads.insert(
                    data1.id(),
                    grad
                )?;
            },
            Self::Cat(datas, dim) => {
                let mut start_idx = 0;
                for data_single in datas {
                    let len = data_single.dim(*dim)?;
                    let grad = node_grad.narrow(*dim, start_idx, len)?;
                    grads.insert(
                        data_single.id(),
                        grad
                    )?;
                    start_idx += len;
                }
            },
        }
        Ok(())
    }
}


// macro_rules! wtensor_float_op {
//     ($ty:ident, $method:ident) => {
//         impl WTensor<$ty> {
//             pub fn sum_exp(&self, dim: u8, broadcast: bool) -> WResult<Self> {
//                 let data = self.read_data()?;
//                 let n = data.dim(dim)?;
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

