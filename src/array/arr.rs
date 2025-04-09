use std::fmt::{Debug, Display};
use super::{dim::WArrDims, data::Data, error::{WError, WResult}};


#[derive(Clone, PartialEq)]
pub struct WArr<T: Data> {
    pub data: T,
    pub dims: WArrDims,
}

impl<T: Data> WArr<T> {
    pub fn new(data: T, dims: WArrDims) -> Self {
        Self { data, dims }
    }
    
    pub fn replace(&mut self, data: Self) -> WResult<()> {
        if data.data.len() != self.data.len() {
            return Err(WError::DimMisMatch(self.data.len(), data.data.len()));
        }
        self.data = data.data;
        Ok(())
    }
}


impl<T: Data> Display for WArr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.to_vec().unwrap();
        write!(f, "WArr {{ Dims: {:?}, {}/{}/{} }}", self.dims.to_vec(), self.dims.dims_len(), self.data.len(), data.len())
    }
}


impl<T: Data> Debug for WArr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.to_vec().unwrap();
        write!(f, "WArr {{ Dims: {:?}, {}/{}/{}, Data: {:?} }}", self.dims.to_vec(), self.dims.dims_len(), self.data.len(), data.len(), data.last())
    }
}
