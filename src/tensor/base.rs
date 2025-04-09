use std::{collections::{HashMap, HashSet}, fmt::{Debug, Display}, io, sync::{atomic::{AtomicBool, AtomicU16, Ordering}, Arc, PoisonError, RwLock, RwLockReadGuard}};

use crate::array::{arr::WArr, data::Data, dim::WArrDims, error::{WError, WResult}};

use super::op::Op;




#[derive(Debug)]
pub struct WTensorInner<T: Data> {
    pub id: u16,
    pub trace: bool,
    pub data: RwLock<WArr<T>>,
    pub op: RwLock<Option<Op<T>>>,
}

impl<T: Data> WTensorInner<T> {
    pub fn new(trace_raw: bool, data: WArr<T>, op: Option<Op<T>>) -> Self {
        let id = new_tensor_id();
        let trace = get_trace(trace_raw);
        let op = if trace {
            op
        } else {
            None
        };
        Self {
            id,
            trace,
            data: RwLock::new(data),
            op: RwLock::new(op),
        }
    }
}


impl<T: Data> Drop for WTensorInner<T> {
    fn drop(&mut self) {
        // log::debug!("tensor {} dropping ...", self.id);
    }
}


pub struct Grad<T: Data>(HashMap<u16, WArr<T>>);


impl<T: Data> From<HashMap<u16, WArr<T>>> for Grad<T> {
    fn from(value: HashMap<u16, WArr<T>>) -> Self {
        Self(value)
    }
}

static ID: AtomicU16 = AtomicU16::new(0);
static TRACE: AtomicBool = AtomicBool::new(true);

fn new_tensor_id() -> u16 {
    let res = ID.fetch_add(1, Ordering::SeqCst) + 1;

    if res == u16::MAX {
        ID.store(4096, Ordering::SeqCst);
    }

    res
}

pub fn disabled_trace() {
    TRACE.store(false, Ordering::SeqCst);
}


fn get_trace(trace_raw: bool) -> bool {
    trace_raw && TRACE.load(Ordering::SeqCst)
}


pub struct WTensor<T: Data>(Arc<WTensorInner<T>>);


#[cfg(feature = "random")]
impl<T: Data> WTensor<T> {
    fn random_inner<S: Into<WArrDims>>(shape: S, uniform: bool) -> WResult<Self> {
        let data = if uniform {
            WArr::random_uniform(shape)?
        } else {
            WArr::random_normal(shape)?
        };
        Ok(Self (Arc::new(WTensorInner::new(true, data, None))))
    }

    pub fn random<S: Into<WArrDims>>(shape: S) -> WResult<Self> {
        Self::random_inner(shape, true)
    }

    pub fn random_normal<S: Into<WArrDims>>(shape: S) -> WResult<Self> {
        Self::random_inner(shape, false)
    }
}

macro_rules! wtensor_from_vec {
    ($from_name0:ident, $from_name1:ident, $ty0:ty, $ty1:ty) => {
        #[allow(clippy::type_complexity)]
        pub fn $from_name0(data: $ty0) -> WResult<Self> {
            let data = WArr::$from_name0(data)?;
            Self::from_data(data, false)
        }

        #[allow(clippy::type_complexity)]
        pub fn $from_name1<V: Into<f64>>(data: $ty1) -> WResult<Self> {
            let data = WArr::<T>::$from_name1(data)?;
            Self::from_data(data, false)
        }
    }
}



impl<T: Data> WTensor<T> {
    pub fn from_op(op: Op<T>) -> WResult<Self> {
        let data = op.forward()?;
        let trace_raw = op.get_nodes().iter().any(| v | v.trace());
        // #[cfg(feature = "logger")]
        // log::debug!("WTensor from op: id {}, trace_raw {}, TRACE {}, trace_final {}", id, trace_raw, TRACE.load(Ordering::SeqCst), trace);
        Ok(Self (Arc::new(WTensorInner::new(trace_raw, data, Some(op)))))
    }

    pub fn from_data(data: WArr<T>, trace: bool) -> WResult<Self> {
        // #[cfg(feature = "logger")]
        // log::debug!("WTensor from data: {}, trace_raw {}, TRACE {}, trace_final {}", id, trace, TRACE.load(Ordering::SeqCst), trace_now);
        Ok(Self (Arc::new(WTensorInner::new(trace, data, None))))
    }

    pub fn ones<S: Into<WArrDims>>(shape: S) -> WResult<Self> {
        let data = WArr::ones(shape)?;
        Self::from_data(data, false)
    }
    
    pub fn like(&self, n: &T::Basic) -> WResult<Self> {
        let data = self.read_data()?.ones_like()?.broadcast_mul(n)?;
        Self::from_data(data, false)
    }
    
    pub fn ones_like(&self) -> WResult<Self> {
        let data = self.read_data()?.ones_like()?;
        Self::from_data(data, false)
    }

    pub fn zeros_like(&self) -> WResult<Self> {
        let data = self.read_data()?.zeros_like()?;
        Self::from_data(data, false)
    }

    pub fn zeros<S: Into<WArrDims>>(shape: S) -> WResult<Self> {
        let data = WArr::zeros(shape)?;
        Self::from_data(data, false)
    }

    // pub fn from_vec<D: Into<WArr<T>>>(data: D, trace: bool) -> WResult<Self> {
    //     let data = data.into();
    //     Self::from_data(data, trace)
    // }
    
    wtensor_from_vec!(from_vec1, from_vec1_f64, Vec<T::Basic>, Vec<V>);
    wtensor_from_vec!(from_vec2, from_vec2_f64, Vec<Vec<T::Basic>>, Vec<Vec<V>>);
    wtensor_from_vec!(from_vec3, from_vec3_f64, Vec<Vec<Vec<T::Basic>>>, Vec<Vec<Vec<V>>>);
    wtensor_from_vec!(from_vec4, from_vec4_f64, Vec<Vec<Vec<Vec<T::Basic>>>>, Vec<Vec<Vec<Vec<V>>>>);
    wtensor_from_vec!(from_vec5, from_vec5_f64, Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>, Vec<Vec<Vec<Vec<Vec<V>>>>>);
    wtensor_from_vec!(from_vec6, from_vec6_f64, Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>);
    wtensor_from_vec!(from_vec7, from_vec7_f64, Vec<Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>>);
    wtensor_from_vec!(from_vec8, from_vec8_f64, Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>>>);

    
    pub fn read_data(&self) -> WResult<RwLockReadGuard<'_, WArr<T>>> {
        Ok(self.0.data.read()?)
    }

    pub fn write_data(&self, data: WArr<T>) -> WResult<()> {
        self.0.data.write()?.replace(data)?;
        Ok(())
    }

    pub fn round(&self, n: T::Basic) -> WResult<()> {
        let data = self.read_data()?.round(n)?;
        self.write_data(data)?;
        Ok(())
    }

    pub fn to_vec(&self) -> WResult<Vec<T::Basic>> {
        self.read_data()?.data.to_vec()
    }
    
    pub fn to_scalar(&self) -> WResult<T::Basic> {
        let mut data = self.to_vec()?;
        if data.len() != 1 {
            return Err(WError::DimMisMatch(0, 0));
        }
        Ok(data.pop().unwrap())
    }

    pub fn dims(&self) -> WResult<WArrDims> {
        let d = self.read_data()?.dims.clone();
        Ok(d)
    }

    pub fn id(&self) -> u16 {
        self.0.id
    }

    pub fn op(&self) -> Option<Op<T>> {
        self.0.op.write().unwrap().take()
    }

    pub fn trace(&self) -> bool {
        self.0.trace
    }

    fn nodes(&self, res: &mut Vec<WTensor<T>>, seen: &mut HashSet<u16>) {
        seen.insert(self.id());
        res.push(self.clone());
        let op = self.0.op.read().unwrap();
        let op = op.as_ref();
        if op.is_none() {
            return;
        }
        for node in op.unwrap().get_nodes() {
            if !node.trace() {
                continue;
            }
            if seen.contains(&node.id()) {
                continue;
            }
            node.nodes(res, seen);
        }
    }

    pub fn get_nodes(&self) -> Vec<WTensor<T>> {
        let mut res = Vec::new();
        let mut seen = HashSet::new();
        self.nodes(&mut res, &mut seen);
        // res.reverse();
        res
    }

    pub fn backward(&self) -> WResult<Grad<T>> {
        let mut grads = Grad::<T>::default();
        grads.insert(self.id(), self.read_data()?.ones_like()?)?;
        let nodes = self.get_nodes();
        #[cfg(feature = "logger")]
        log::debug!("backward, get nodes: {:?}", nodes.iter().map(| v | v.id()).collect::<Vec<_>>());
        // 计算梯度
        for node in nodes.iter() {
            log::debug!("backward, calc node: {}", node);
            if !node.trace() {
                continue;
            }
            let op = node.op();
            if op.is_none() {
                continue;
            }
            let node_id = node.id();
            let node_data = node.read_data()?;
            op.unwrap().backward(&mut grads, node_id, &node_data)?;
        }
        Ok(grads)
    }
}



impl<T: Data> Display for WTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.read_data().unwrap();
        write!(f, "WTensor {{ id: {}, trace: {}, dim: {:?}, data: {} }}", self.id(), self.trace(), data.dims, data)
    }
}

impl<T: Data> Debug for WTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.read_data().unwrap();
        write!(f, "WTensor {{ id: {}, trace: {}, dim: {:?}, data: {:?} }}", self.id(), self.trace(), data.dims, data)
    }
}


impl<T: Data> Clone for WTensor<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: Data> PartialEq for WTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        let data0 = self.read_data().unwrap();
        let data1 = other.read_data().unwrap();
        data0.eq(&data1)
    }
}


impl From<io::Error> for WError {
    fn from(value: io::Error) -> Self {
        Self::Read(format!("IO: {}", value))
    }
}

impl<I> From<PoisonError<I>> for WError {
    fn from(value: PoisonError<I>) -> Self {
        Self::Read(format!("Lock: {}", value))
    }
}


impl<T: Data> Default for Grad<T> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}


impl<T: Data> Grad<T> {
    pub fn insert(&mut self, id: u16, data: WArr<T>) -> WResult<()> {
        if let Some(data_old) = self.0.get_mut(&id) {
            let data = data_old.add(&data)?;
            #[cfg(feature = "logger")]
            log::debug!("Grad {} {} {}", id, data, data_old);
            data_old.replace(data)?;
        } else {
            #[cfg(feature = "logger")]
            log::debug!("Grad {} {}", id, data);
            self.0.insert(id, data);
        };
        Ok(())
    }

    pub fn remove(&mut self, id: u16) -> Option<WArr<T>> {
        self.0.remove(&id)
    }

    pub fn get(&mut self, id: u16) -> Option<&WArr<T>> {
        self.0.get(&id)
    }

    pub fn keys(&self) -> Vec<u16> {
        self.0.keys().cloned().collect::<Vec<_>>()
    }
}



