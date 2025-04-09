
use std::{fmt::{Debug, Display}, vec::IntoIter};

use crate::array::{data::Data, error::{WError, WResult}};

use super::error::CudaError;

#[repr(C)]
pub struct F32 {
    data: *const f32,
    len: usize,
}

#[link(name = "f32")]
unsafe extern "C" {
    fn wdata_new(data: *const f32, len: usize) -> *mut F32;
    fn wdata_new_with(data: *const F32, len: usize, len_all: usize) -> *mut F32;
    fn wdata_clone(data: *const F32) -> *mut F32;
    fn wdata_drop(obj: *mut F32);
    fn wdata_eq(lhs: *const F32, rhs: *const F32) -> bool;
    fn wdata_basic_op(lhs: *const F32, rhs: *const F32, op: u8) -> *mut F32;
    fn wdata_broadcast_op(lhs: *const F32, rhs: f32, op: u8) -> *mut F32;
    fn wdata_map_op(lhs: *const F32, op: u8) -> *mut F32;
    fn wdata_get_data_single(obj: *const F32, index: usize) -> f32;
    fn wdata_get_data_with_index(obj: *const F32, index: *const usize, len: usize) -> *mut f32;
    fn wdata_get_data_all(obj: *const F32) -> *mut f32;
    fn wdata_dim_op(obj: *const F32, dim0: usize, n0: usize, n1: usize, op: u8) -> *mut F32;
    fn wdata_transpose(obj: *const F32, dim0: usize, dim1: usize, n0: usize, n1: usize) -> *mut F32;
    fn wdata_concat(lhs: *const F32, rhs: *const F32, dim0: usize, dim1: usize, n0: usize, n1: usize) -> *mut F32;
    fn wdata_broadcast(obj: *const F32, n: usize, n0: usize, n1: usize) -> *mut F32;
    fn wdata_where_cond(obj: *const F32, s: f32, a: f32, b: f32) -> *mut F32;
    fn wdata_matmul_1d(lhs: *const F32, rhs: *const F32) -> *mut F32;
    fn wdata_matmul_2d(lhs: *const F32, rhs: *const F32, n: usize, n0: usize, n1: usize) -> *mut F32;
    fn wdata_matmul_nd(lhs: *const F32, rhs: *const F32, n: usize, n0: usize, n1: usize, pre: usize) -> *mut F32;
    fn device_set(dev: u8);
    fn device_error() -> u16;
    fn device_reset();
}



#[allow(dead_code)]
pub fn cuda_device_set(dev: u8) {
    unsafe {
        device_set(dev);
    }
}

#[allow(dead_code)]
pub fn cuda_device_reset() {
    unsafe {
        device_reset();
    }
}


pub fn cuda_device_error() -> CudaError {
    let err = unsafe {
        device_error()
    };
    CudaError::from(err)
}



impl F32 {
    pub fn from_slice_date(data: &[F32]) -> Self {
        let len_all = data.iter().map(| v | v.len).sum();
        unsafe {
            let ptr = wdata_new_with(data.as_ptr(), data.len(), len_all);
            *Box::from_raw(ptr)
        }
    }

    pub fn from_slice(data: &[f32]) -> Self {
        unsafe {
            let ptr = wdata_new(data.as_ptr(), data.len());
            *Box::from_raw(ptr)
        }
    }

    pub fn basic_op(&self, rhs: &Self, op: u8) -> Self {
        unsafe {
            let ptr_res = wdata_basic_op(self as *const _, rhs as *const _, op);
            *Box::from_raw(ptr_res)
        }
    }

    pub fn broadcast_op(&self, rhs: f32, op: u8) -> Self {
        unsafe {
            let ptr_res = wdata_broadcast_op(self as *const _, rhs, op);
            *Box::from_raw(ptr_res)
        }
    }

    pub fn map_op(&self, op: u8) -> Self {
        unsafe {
            let ptr_res = wdata_map_op(self as *const _, op);
            *Box::from_raw(ptr_res)
        }
    }

    pub fn get_data_single(&self, index: usize) -> f32 {
        unsafe {
            wdata_get_data_single(self as *const _, index)
        }
    }

    pub fn get_data_with_index(&self, index: Vec<usize>) -> Vec<f32> {
        unsafe {
            let ptr_res = wdata_get_data_with_index(self as *const _, index.as_ptr(), index.len());
            Vec::from_raw_parts(ptr_res, index.len(), index.len())
        }
    }
    
    pub fn get_data_all(&self) -> Vec<f32> {
        unsafe {
            let ptr_res = wdata_get_data_all(self as *const _);
            Vec::from_raw_parts(ptr_res, self.len, self.len)
        }
    }

    pub fn clone_new(&self) -> Self {
        unsafe {
            let ptr_res = wdata_clone(self as *const _);
            *Box::from_raw(ptr_res)
        }
    }

    pub fn eq_all(&self, rhs: &Self) -> bool {
        unsafe {
            wdata_eq(self as *const _, rhs as *const _)
        }
    }


    fn map_dim_data(&self, dims: &[usize], dim: usize, gap: usize) -> (usize, usize, usize) {
        let dim0 = dims[dim];
        let n0 = dims.iter().copied().take(dim).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim+gap).reduce(| a, b | a * b).unwrap_or(1);
        (dim0, n0, n1)
    }


    fn map_dim_data2(&self, dims: &[usize], dim0: usize, dim1: usize) -> (usize, usize, usize, usize) {
        let n_dim0 = dims[dim0];
        let n_dim1 = dims[dim1];
        let n0 = dims.iter().copied().take(dim0).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim1+1).reduce(| a, b | a * b).unwrap_or(1);
        (n_dim0, n_dim1, n0, n1)
    }

    fn map_dim_inner(&self, dim0: usize, n0: usize, n1: usize, op: u8) -> Self {
        unsafe {
            let ptr_res = wdata_dim_op(self as *const _, dim0, n0, n1, op);
            *Box::from_raw(ptr_res)
        }
    }

    fn where_cond_inner(&self, s: f32, a: f32, b: f32) -> Self {
        unsafe {
            let ptr_res = wdata_where_cond(self as *const _, s, a, b);
            *Box::from_raw(ptr_res)
        }
    }

    fn transpose_inner(&self, dim0: usize, dim1: usize, n0: usize, n1: usize) -> Self {
        unsafe {
            let ptr_res = wdata_transpose(self as *const _, dim0, dim1, n0, n1);
            *Box::from_raw(ptr_res)
        }
    }

    fn concat_inner(&self, rhs: &Self, dim0: usize, dim1: usize, n0: usize, n1: usize) -> Self {
        unsafe {
            let ptr_res = wdata_concat(self as *const _, rhs as *const _, dim0, dim1, n0, n1);
            *Box::from_raw(ptr_res)
        }
    }

    fn broadcast_inner(&self, n: usize, n0: usize, n1: usize) -> Self {
        unsafe {
            let ptr_res = wdata_broadcast(self as *const _, n, n0, n1);
            *Box::from_raw(ptr_res)
        }
    }

    fn matmul_1d_inner(&self, rhs: &Self) -> Self {
        unsafe {
            let ptr_res = wdata_matmul_1d(self as *const _, rhs as *const _);
            *Box::from_raw(ptr_res)
        }
    }

    fn matmul_2d_inner(&self, rhs: &Self, n: usize, n0: usize, n1: usize) -> Self {
        unsafe {
            let ptr_res = wdata_matmul_2d(self as *const _, rhs as *const _, n, n0, n1);
            *Box::from_raw(ptr_res)
        }
    }

    fn matmul_nd_inner(&self, rhs: &Self, n: usize, n0: usize, n1: usize, pre: usize) -> Self {
        unsafe {
            let ptr_res = wdata_matmul_nd(self as *const _, rhs as *const _, n, n0, n1, pre);
            *Box::from_raw(ptr_res)
        }
    }

    fn check_error(&self) -> WResult<()> {
        if self.len == 0 {
            let err = cuda_device_error();
            return Err(WError::Cuda(format!("{:?}", err)));
        }
        Ok(())
    }

    fn drop_inner(&mut self){
        unsafe {
            wdata_drop(self as *mut _);
        }
    }
}

impl Drop for F32 {
    fn drop(&mut self) {
        self.drop_inner();
    }
}











macro_rules! data_methods {
    ($method:ident, $method_broadcast:ident, $op:expr) => {
        fn $method(&self, rhs: &Self) -> WResult<Self> {
            let res = self.basic_op(rhs, $op);
            res.check_error()?;
            Ok(res)
        }

        fn $method_broadcast(&self, rhs: &Self::Basic) -> WResult<Self> {
            let res = self.broadcast_op(*rhs, $op);
            res.check_error()?;
            Ok(res)
        }
    };

    ($method:ident, $op:expr, "single") => {
        fn $method(&self) -> WResult<Self> {
            let res = self.map_op($op);
            res.check_error()?;
            Ok(res)
        }
    };

    ($method:ident, $op:expr, "dim") => {
        fn $method(&self, dims: &[usize], dim: usize) -> WResult<Self> {
            let (dim0, n0, n1) = self.map_dim_data(dims, dim, 1);
            let res = self.map_dim_inner(dim0, n0, n1, $op);
            res.check_error()?;
            Ok(res)
        }
    };
}


impl Data for F32 {
    type Basic = f32;
    type BasicIntoInterator = IntoIter<Self::Basic>;
    
    fn zero(&self) -> Self::Basic {
        0.0
    }

    fn one(&self) -> Self::Basic {
        1.0
    }
    
    fn len(&self) -> usize {
        self.len
    }
    
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    fn neg_one(&self) -> Self::Basic {
        -1.0
    }
    
    fn to_vec(&self) -> WResult<Vec<Self::Basic>> {
        let data = self.get_data_all();
        Ok(data)
    }
    
    fn f64_to_basic(data: f64) -> WResult<Self::Basic> {
        Ok(data as Self::Basic)
    }
    
    fn usize_to_basic(data: usize) -> WResult<Self::Basic> {
        Ok(data as Self::Basic)
    }
    
    fn from_vec(data: Vec<Self::Basic>) -> WResult<Self> {
        let res = Self::from_slice(&data);
        res.check_error()?;
        Ok(res)
    }
    
    fn from_vec_data(data: Vec<Self>) -> WResult<Self> {
        let res = Self::from_slice_date(&data);
        res.check_error()?;
        Ok(res)
    }
    
    data_methods!(exp, 0, "single");
    data_methods!(log2, 1, "single");
    data_methods!(ln, 2, "single");
    data_methods!(neg, 3, "single");
    data_methods!(abs, 4, "single");
    data_methods!(relu, 5, "single");
    
    
    fn round(&self, n: Self::Basic) -> WResult<Self> {
        let res = self.broadcast_op(n, 4);
        res.check_error()?;
        Ok(res)
    }

    data_methods!(add, broadcast_add, 0);
    data_methods!(sub, broadcast_sub, 1);
    data_methods!(mul, broadcast_mul, 2);
    data_methods!(div, broadcast_div, 3);
    
    
    fn eq_item(&self, rhs: &Self) -> WResult<Self> {
        let res = self.basic_op(rhs, 4);
        res.check_error()?;
        Ok(res)
    }
    
    fn map_item<F: Fn(&Self::Basic) -> Self::Basic>(&self, f: F) -> WResult<Self> {
        let data_raw = self.get_data_all();
        let data = data_raw.iter().map(f).collect::<Vec<_>>();
        Self::from_vec(data)
    }
    
    fn map_dim<F: Fn(Self::BasicIntoInterator) -> Self::Basic>(&self, dims: &[usize], dim: usize, f: F) -> WResult<Self> {
        let data_raw = self.get_data_all();

        let (dim0, n0, n1) = self.map_dim_data(dims, dim, 1);
        
        let mut data = Vec::with_capacity(n0 * n1);
        for n0_temp in 0..n0 {
            for n1_temp in 0..n1 {
                let i_start0 = n0_temp * n1 * dim0;
                let data_temp = (0..dim0).map(| i |{
                    let i_start = i * n1 + i_start0;
                    let index = n1_temp + i_start;
                    data_raw[index]
                }).collect::<Vec<_>>().into_iter();
                let t = f(data_temp);
                data.push(t);
            }
        }
        Self::from_vec(data)
    }
    
    fn item_iter(&self) -> Self::BasicIntoInterator {
        self.get_data_all().into_iter()
    }
    
    fn matmul_1d(&self, rhs: &Self) -> WResult<Self> {
        let res = self.matmul_1d_inner(rhs);
        res.check_error()?;
        Ok(res)
    }

    fn matmul_2d(&self, rhs: &Self, n0: usize, n1: usize, n: usize) -> WResult<Self> {
        let res = self.matmul_2d_inner(rhs, n, n0, n1);
        res.check_error()?;
        Ok(res)
    }

    fn matmul_nd(&self, rhs: &Self, n0: usize, n1: usize, n: usize, pre: usize) -> WResult<Self> {
        let res = self.matmul_nd_inner(rhs, n, n0, n1, pre);
        res.check_error()?;
        Ok(res)
    }
    
    

    data_methods!(mean, 0, "dim");
    data_methods!(sum, 0, "dim");
    data_methods!(max, 0, "dim");
    data_methods!(min, 0, "dim");
    
    
    fn where_cond(&self, s: &Self::Basic, a: &Self::Basic, b: &Self::Basic) -> WResult<Self> {
        let res = self.where_cond_inner(*s, *a, *b);
        res.check_error()?;
        Ok(res)
    }
    
    fn transpose(&self, dims: &[usize], dim0: usize, dim1: usize) -> WResult<Self> {
        let (n_dim0, n_dim1, n0, n1) = self.map_dim_data2(dims, dim0, dim1);
        let res = self.transpose_inner(n_dim0, n_dim1, n0, n1);
        Ok(res)
    }
    
    fn concat(&self, dims: &[usize], rhs: &Self, dim: usize, n: usize) -> WResult<Self> {
        let (dim0, n0, n1) = self.map_dim_data(dims, dim, 1);
        let res = self.concat_inner(rhs, dim0, n, n0, n1);
        res.check_error()?;
        Ok(res)
    }
    
    fn stack(&self, dims: &[usize], rhs: &Self, dim: usize) -> WResult<Self> {
        let (_, n0, n1) = self.map_dim_data(dims, dim, 0);
        let res = self.concat_inner(rhs, 1, 1, n0, n1);
        res.check_error()?;
        Ok(res)
    }
    
    fn broadcast(&self, dims: &[usize], dim: usize, n: usize) -> WResult<Self> {
        let (_, n0, n1) = self.map_dim_data(dims, dim, 1);
        let res = self.broadcast_inner(n, n0, n1);
        res.check_error()?;
        Ok(res)
    }

    fn conv2d(&self, kernel: &Self, params: &crate::array::data::ParamsConv2D) -> WResult<Self> {
        todo!()
    }

    fn flipped(&self, dims: &[usize], flip: &[usize]) -> WResult<Self> {
        todo!()
    }

    fn conv_transpose2d(&self, kernel: &Self, params: &crate::array::data::ParamsConvTranspose2D) -> WResult<Self> {
        todo!()
    }
}


impl Debug for F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F32 {{ len: {}, data: {:?} }}", self.len, self.get_data_all())
    }
}

impl Display for F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F32 {{ len: {} }}", self.len)
    }
}

impl PartialEq for F32 {
    fn eq(&self, other: &Self) -> bool {
        self.eq_all(other)
    }
}

impl Clone for F32 {
    fn clone(&self) -> Self {
        self.clone_new()
    }
}



