
use std::{fmt::{Debug, Display}, vec::IntoIter};

use super::error::{WError, WResult};


#[derive(Debug)]
pub struct ParamsConv2D {
    pub batch_size: usize,
    pub data_w: usize,
    pub data_h: usize,
    pub kernel_w: usize,
    pub kernel_h: usize,
    pub padding_w: usize,
    pub padding_h: usize,
    pub stride_w: usize,
    pub stride_h: usize,
    pub dilation_w: usize,
    pub dilation_h: usize,
    pub groups: usize,
    pub channels_out: usize,
    pub channels_in: usize
}

impl ParamsConv2D {
    pub fn get_output_dims(&self) -> WResult<[usize; 4]> {
        let e_kernel_h = (self.kernel_h - 1) * self.dilation_h + 1;
        let e_kernel_w = (self.kernel_w - 1) * self.dilation_w + 1;

        let h_max = match (self.data_h + self.padding_h * 2).checked_sub(e_kernel_h - 1) {
            Some(v) => v,
            None => return Err(WError::DimNumError("dilation or kernel_size too big".to_string())),
        };
        let w_max = match (self.data_w + self.padding_w * 2).checked_sub(e_kernel_w - 1) {
            Some(v) => v,
            None => return Err(WError::DimNumError("dilation or kernel_size too big".to_string())),
        };
        
        let h_n = (h_max - 1) / self.stride_h + 1;
        let w_n = (w_max - 1) / self.stride_w + 1;

        Ok([self.batch_size, self.groups * self.channels_out, h_n, w_n])
    }
}



pub struct ParamsConvTranspose2D {
    pub batch_size: usize,
    pub data_w: usize,
    pub data_h: usize,
    pub kernel_w: usize,
    pub kernel_h: usize,
    pub padding_w: usize,
    pub padding_h: usize,
    pub output_padding_w: usize,
    pub output_padding_h: usize,
    pub stride_w: usize,
    pub stride_h: usize,
    pub dilation_w: usize,
    pub dilation_h: usize,
    pub groups: usize,
    pub channels_out: usize,
    pub channels_in: usize
}

impl ParamsConvTranspose2D {
    pub fn get_data_hw(&self) -> WResult<(usize, usize, usize, usize)> {
        let padding_h = self.kernel_h - self.padding_h - 1;
        let padding_w = self.kernel_w - self.padding_w - 1;
        let data_h = self.stride_h * self.data_h + padding_h * 2;
        let data_w = self.stride_w * self.data_w + padding_w * 2;
        Ok((data_h, data_w, padding_h, padding_w))
    }

    pub fn to_params_conv2d(&self) -> ParamsConv2D {
        let (data_h, data_w, _, _) = self.get_data_hw().unwrap();
        ParamsConv2D {
            batch_size: self.batch_size,
            data_w,
            data_h,
            kernel_w: self.kernel_w,
            kernel_h: self.kernel_h,
            padding_w: self.output_padding_w,
            padding_h: self.output_padding_h,
            stride_w: 1,
            stride_h: 1,
            dilation_w: self.dilation_w,
            dilation_h: self.dilation_h,
            groups: self.groups,
            channels_out: self.channels_out,
            channels_in: self.channels_in,
        }
    }

    pub fn get_output_dims(&self) -> WResult<[usize; 4]> {
        self.to_params_conv2d().get_output_dims()
    }
}




pub trait Data
where
    Self: Sized + Clone + PartialEq + Debug,
    Self::Basic: Debug + PartialEq + Display + Clone
{
    type Basic;

    // prop
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn one(&self) -> Self::Basic;
    fn zero(&self) -> Self::Basic;
    fn neg_one(&self) -> Self::Basic;

    // gen
    fn to_vec(&self) -> WResult<Vec<Self::Basic>>;
    fn f64_to_basic(data: f64) -> WResult<Self::Basic>;
    fn usize_to_basic(data: usize) -> WResult<Self::Basic>;
    fn from_vec(data: Vec<Self::Basic>) -> WResult<Self>;
    fn from_vec_data(data: Vec<Self>) -> WResult<Self>;
    
    // float
    fn sqrt(&self) -> WResult<Self>;
    fn exp(&self) -> WResult<Self>;
    fn log2(&self) -> WResult<Self>;
    fn ln(&self) -> WResult<Self>;
    fn round(&self, n: Self::Basic) -> WResult<Self>;

    // basic
    fn add(&self, rhs: &Self) -> WResult<Self>;
    fn sub(&self, rhs: &Self) -> WResult<Self>;
    fn div(&self, rhs: &Self) -> WResult<Self>;
    fn mul(&self, rhs: &Self) -> WResult<Self>;
    fn eq_item(&self, rhs: &Self) -> WResult<Self>;
    fn neg(&self) -> WResult<Self>;

    // broadcast
    fn broadcast_add(&self, rhs: &Self::Basic) -> WResult<Self>;
    fn broadcast_sub(&self, rhs: &Self::Basic) -> WResult<Self>;
    fn broadcast_div(&self, rhs: &Self::Basic) -> WResult<Self>;
    fn broadcast_mul(&self, rhs: &Self::Basic) -> WResult<Self>;
    
    // normal without dim
    fn abs(&self) -> WResult<Self>;
    fn relu(&self) -> WResult<Self>;
    fn map_item<F: Fn(&Self::Basic) -> Self::Basic>(&self, f: F) -> WResult<Self>;
    fn map_dim<F: Fn(IntoIter<Self::Basic>) -> Self::Basic>(&self, dims: &[usize], dim: usize, f: F) -> WResult<Self>;
    fn item_iter(&self) -> IntoIter<Self::Basic>;

    // normal with dim
    fn matmul_1d(&self, rhs: &Self) -> WResult<Self>;
    fn matmul_2d(&self, rhs: &Self, n0: usize, n1: usize, n: usize) -> WResult<Self>;
    fn matmul_nd(&self, rhs: &Self, n0: usize, n1: usize, n: usize, pre: usize) -> WResult<Self>;
    fn mean(&self, dims: &[usize], dim: usize) -> WResult<Self>;
    fn sum(&self, dims: &[usize], dim: usize) -> WResult<Self>;
    fn max(&self, dims: &[usize], dim: usize) -> WResult<Self>;
    fn min(&self, dims: &[usize], dim: usize) -> WResult<Self>;
    fn where_cond(&self, s: &Self::Basic, a: &Self::Basic, b: &Self::Basic) -> WResult<Self>;
    fn transpose(&self, dims: &[usize], dim0: usize, dim1: usize) -> WResult<Self>;
    fn concat(&self, dims: &[usize], rhs: &Self, dim: usize, n: usize) -> WResult<Self>;
    fn stack(&self, dims: &[usize], rhs: &Self, dim: usize) -> WResult<Self>;
    fn broadcast(&self, dims: &[usize], dim: usize, n: usize) -> WResult<Self>;
    fn conv2d(&self, kernel: &Self, params: &ParamsConv2D) -> WResult<Self>;
    fn flipped(&self, dims: &[usize], flip: &[usize]) -> WResult<Self>;
    fn conv_transpose2d(&self, kernel: &Self, params: &ParamsConvTranspose2D) -> WResult<Self>;

    // select
    fn gather(&self, indexes: &Self, dims_self: &[usize], dims_ids: &[usize], dim: usize) -> WResult<Self>;
    fn scatter(&self, indexes: &Self, src: &Self, dims_self: &[usize], dims_ids: &[usize], dim: usize) -> WResult<Self>;
}

