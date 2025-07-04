
use std::{ops::Deref, vec::IntoIter};

use crate::array::{data::{Data, ParamsConv2D, ParamsConvTranspose2D}, error::WResult};

use crate::dtype::utils::DataStep;



#[derive(Debug, PartialEq, Clone)]
pub struct Boolean {
    data: Vec<bool>,
    threhold: usize
}

impl Boolean {
    pub fn new(data: Vec<bool>) -> WResult<Self> {
        Ok(Self(data))
    }

    pub fn map_zip_item<F: Fn((&bool, &bool)) -> bool>(&self, rhs: &Self, f: F) -> WResult<Self> {
        let res = self.data.iter().zip(rhs.data.iter()).map(f).collect::<Vec<_>>();
        Self::new(res)
    }
}

impl Deref for Boolean {
    type Target = Vec<bool>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}



impl Data for Boolean {
    type Basic = bool;

    // prop
    fn len(&self) -> usize {
        self.data.len()
    }
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    fn one(&self) -> Self::Basic {
        true
    }
    fn zero(&self) -> Self::Basic {
        false
    }
    fn neg_one(&self) -> Self::Basic {
        todo!()
    }

    // gen
    fn to_vec(&self) -> WResult<Vec<Self::Basic>> {
        Ok(self.clone().data)
    }

    fn f64_to_basic(data: f64) -> WResult<Self::Basic> {
        Ok(data > 0)
    }

    fn usize_to_basic(data: usize) -> WResult<Self::Basic> {
        Ok(data > 0)
    }

    fn from_vec(data: Vec<Self::Basic>) -> WResult<Self> {
        Self::new(data)
    }

    fn from_vec_data(data: Vec<Self>) -> WResult<Self> {
        let data_now = data.into_iter().reduce(| mut a, b | {
            a.data.extend(b.data);
            a
        }).unwrap();
        Ok(data_now)
    }
    
    // float
    fn exp(&self) -> WResult<Self> {
        todo!()
    }
    
    fn log2(&self) -> WResult<Self> {
        todo!()
    }

    fn ln(&self) -> WResult<Self> {
        todo!()
    }

    fn round(&self, n: Self::Basic) -> WResult<Self> {
        todo!()
    }

    // basic
    fn add(&self, rhs: &Self) -> WResult<Self> {
        todo!()
    }

    fn sub(&self, rhs: &Self) -> WResult<Self> {
        todo!()
    }

    fn div(&self, rhs: &Self) -> WResult<Self> {
        todo!()
    }

    fn mul(&self, rhs: &Self) -> WResult<Self> {
        todo!()
    }

    fn eq_item(&self, rhs: &Self) -> WResult<Self> {
        let one = self.one();
        let zero = self.zero();
        self.map_zip_item(rhs, | (a, b) | {
            if a == b {
                one
            } else {
                zero
            }
        })
    }

    fn neg(&self) -> WResult<Self> {
        todo!()
    }
    
    // broadcast
    fn broadcast_add(&self, rhs: &Self::Basic) -> WResult<Self> {
        todo!()
    }

    fn broadcast_sub(&self, rhs: &Self::Basic) -> WResult<Self> {
        todo!()
    }

    fn broadcast_div(&self, rhs: &Self::Basic) -> WResult<Self> {
        todo!()
    }

    fn broadcast_mul(&self, rhs: &Self::Basic) -> WResult<Self> {
        todo!()
    }
    

    // normal without dim
    fn abs(&self) -> WResult<Self> {
        todo!()
    }

    fn relu(&self) -> WResult<Self> {
        let zero = self.zero();
        self.map_item(| v | {
            if v > &zero {
                *v
            } else {
                zero
            }
        })
    }

    fn map_item<F: Fn(&bool) -> bool>(&self, f: F) -> WResult<Self> {
        let res = self.data.iter().map(f).collect::<Vec<_>>();
        Self::new(res)
    }

    fn map_dim<F: Fn(IntoIter<bool>) -> bool>(&self, dims: &[usize], dim: usize, f: F) -> WResult<Self> {
        let dim0 = dims[dim];
        let n0 = dims.iter().copied().take(dim).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim+1).reduce(| a, b | a * b).unwrap_or(1);
        let mut data = Vec::with_capacity(n0 * n1);
        for n0_temp in 0..n0 {
            for n1_temp in 0..n1 {
                let i_start0 = n0_temp * n1 * dim0;
                let data_temp = (0..dim0).map(| i |{
                    let i_start = i * n1 + i_start0;
                    let index = n1_temp + i_start;
                    self.data[index]
                }).collect::<Vec<_>>().into_iter();
                let t = f(data_temp);
                data.push(t);
            }
        }
        Self::from_vec(data)
    }
    
    fn item_iter(&self) -> IntoIter<Self::Basic> {
        self.data.clone().into_iter()
    }

    // normal with dim
    fn matmul_1d(&self, rhs: &Self) -> WResult<Self> {
        let data = self.mul(rhs)?.data.iter().sum();
        Self::from_vec(vec![data])
    }

    fn matmul_2d(&self, rhs: &Self, n0: usize, n1: usize, n: usize) -> WResult<Self> {
        let res = (0..n0).flat_map(| i | {
            let i0 = i*n;
            (0..n1).map(move | j |{
                (0..n).map(move | i |{
                    let a = self.data[i0 + i];
                    let b = rhs.data[j + i * n1];
                    a * b
                }).sum()
            })
        }).collect::<Vec<_>>();
        Self::from_vec(res)
    }

    fn matmul_nd(&self, rhs: &Self, n0: usize, n1: usize, n: usize, pre: usize) -> WResult<Self> {
        let u1 = n0 * n;
        let u2 = n * n1;
        let res = (0..pre).flat_map(| s |{
            let start1 = s * u1;
            let start2 = s * u2;
            (0..n0).flat_map(move | i | {
                // println!("{} {} {} {}", start1, i, self.data.len(), i*n + start1);
                let data_step1 = DataStep::new(&self.data, 1, i*n + start1);
                (0..n1).map(move | j |{
                    let data_step2 = DataStep::new(&rhs.data, n1, j + start2);
                    // println!("gap");
                    data_step1.take(n).zip(data_step2).map(| (a, b) | {
                        // println!("{:?} {:?}", a, b);
                        a * b
                    }).sum()
                })
            })
        }).collect::<Vec<_>>();
        
        Self::from_vec(res)
    }

    fn mean(&self, dims: &[usize], dim: usize) -> WResult<Self> {
        todo!()
    }

    fn sum(&self, dims: &[usize], dim: usize) -> WResult<Self> {
        self.map_dim(dims, dim, | v | v.sum())
    }

    fn max(&self, dims: &[usize], dim: usize) -> WResult<Self> {
        self.map_dim(dims, dim, | v | v.reduce(| a, b | {
            if a > b {
                a
            } else {
                b
            }
        }).unwrap())
    }

    fn min(&self, dims: &[usize], dim: usize) -> WResult<Self> {
        self.map_dim(dims, dim, | v | v.reduce(| a, b | {
            if a > b {
                b
            } else {
                a
            }
        }).unwrap())
    }
    
    fn where_cond(&self, s: &bool, a: &bool, b: &bool) -> WResult<Self> {
        self.map_item(| v | {
            if v >= s {
                *a
            } else {
                *b
            }
        })
    }

    fn transpose(&self, dims: &[usize], dim0: usize, dim1: usize) -> WResult<Self> {
        let n0 = dims.iter().copied().take(dim0).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim1+1).reduce(| a, b | a * b).unwrap_or(1);
        let n_dim0 = dims[dim0];
        let n_dim1 = dims[dim1];
        
        let mut data_res = vec![self.one(); self.data.len()];
        for n0_temp in 0..n0 {
            let n0_temp = n0_temp * n_dim0 * n_dim1 * n1;
            for n1_temp in 0..n1 {
                for i in 0..n_dim0 {
                    for j in 0..n_dim1 {
                        let a0 = i * n_dim1 * n1 + j * n1;
                        let b0 = j * n_dim0 * n1 + i * n1;
                        let a = a0 + n1_temp + n0_temp;
                        let b = b0 + n1_temp + n0_temp;
                        data_res[b] = self.data[a];
                    }
                }
            }
        }

        Self::from_vec(data_res)
    }


    fn concat(&self, dims: &[usize], rhs: &Self, dim: usize, n: usize) -> WResult<Self> {
        let dim0 = dims[dim];
        let n0 = dims.iter().copied().take(dim).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim+1).reduce(| a, b | a * b).unwrap_or(1);
        let dim_sum = dim0 + n;

        let mut data = Vec::with_capacity(n0 * dim_sum * n1);
        for n0_temp in 0..n0 {
            let n0_start = n0_temp * n1;
            let n0_start0 = n0_start * dim0;
            let n0_start1 = n0_start * n;
            for i in 0..dim_sum {
                let (is_first, i_start) = if i < dim0 {
                    (true, i * n1 + n0_start0)
                } else {
                    (false, (i - dim0) * n1 + n0_start1)
                };
                for n1_temp in 0..n1 {
                    let index = n1_temp + i_start;
                    let v = if is_first {
                        self.data[index]
                    } else {
                        rhs.data[index]
                    };
                    data.push(v);
                }
            }
        }

        Self::from_vec(data)
    }

    fn stack(&self, dims: &[usize], rhs: &Self, dim: usize) -> WResult<Self> {
        let n0 = dims.iter().copied().take(dim).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim).reduce(| a, b | a * b).unwrap_or(1);
        
        let mut data = Vec::with_capacity(n0 * 2 * n1);
        for n0_temp in 0..n0 {
            let n_start = n0_temp * n1;
            for i in 0..2 {
                let is_first = i == 0;
                for n1_temp in 0..n1 {
                    let index = n1_temp + n_start;
                    let v = if is_first {
                        self.data[index]
                    } else {
                        rhs.data[index]
                    };
                    data.push(v);
                }
            }
        }

        Self::from_vec(data)
    }
    
    fn broadcast(&self, dims: &[usize], dim: usize, n: usize) -> WResult<Self> {
        let n0 = dims.iter().copied().take(dim).reduce(| a, b | a * b).unwrap_or(1);
        let n1 = dims.iter().copied().skip(dim+1).reduce(| a, b | a * b).unwrap_or(1);
        
        let mut data = Vec::with_capacity(n0 * n * n1);
        for n0_temp in 0..n0 {
            let i_start = n0_temp * n1;
            for _ in 0..n {
                for n1_temp in 0..n1 {
                    let index = n1_temp + i_start;
                    let v = self.data[index];
                    data.push(v);
                }
            }
        }

        Self::from_vec(data)
    }

    // https://github.com/huggingface/candle/blob/main/candle-core/src/conv.rs
    fn conv2d(&self, kernel: &Self, params: &ParamsConv2D) -> WResult<Self> {
        let [_, _, h_n, w_n] = params.get_output_dims()?;
        let len_res = params.batch_size * params.groups * params.channels_out * h_n * w_n;
        // println!("len_res {} {} {} {}, {}", params.data_h, params.padding_h, params.stride_h, h_n, len_res);
        let mut res = vec![false; len_res];
        
        let l = params.channels_in * params.kernel_h * params.kernel_w;
        let mut i_self = vec![0; l];
        let mut i_kernel = vec![0; l];
        let mut i_i;
        let mut i_i_self;
        let mut i_i_self_h;
        let mut i_i_self_w;
        let mut i_i_kernel;
        let mut i_res;

        let ns_res = [
            params.groups * params.channels_out * h_n * w_n,
            params.channels_out * h_n * w_n,
            h_n * w_n,
            w_n,
        ];

        let ns_i = [
            params.kernel_h * params.kernel_w,
            params.kernel_w,
        ];

        let ns_i_self = [
            params.groups * params.channels_in * params.data_h * params.data_w,
            params.channels_in * params.data_h * params.data_w,
            params.data_h * params.data_w,
            params.data_w,
        ];

        let ns_i_kernel = [
            params.channels_in * params.kernel_h * params.kernel_w,
            params.kernel_h * params.kernel_w,
            params.kernel_w,
        ];
        let max_h = params.data_h + params.padding_h;
        let max_w = params.data_w + params.padding_w;
        
        // dim self: batch_size, groups * channels_in, data_h, data_w
        // dim kernel: channels_out, channels_in, data_h, data_w
        // res: params.batch_size, params.groups, params.channels_out, h_n, 
        for b_i in 0..params.batch_size {
            for h_i in 0..h_n {
                for w_i in 0..w_n {
                    for g_i in 0..params.groups {
                        for c_i in 0..params.channels_out {
                            // reset
                            for c_ii in 0..params.channels_in {
                                for h_ii in 0..params.kernel_h {
                                    for w_ii in 0..params.kernel_w {
                                        i_i = c_ii * ns_i[0] + h_ii * ns_i[1] + w_ii;
                                        i_i_self_h = h_i * params.stride_h + h_ii * params.dilation_h;
                                        i_i_self_w = w_i * params.stride_w + w_ii * params.dilation_w;
                                        if i_i_self_h < params.padding_h || i_i_self_w < params.padding_w || i_i_self_h >= max_h || i_i_self_w >= max_w {
                                            i_i_self = usize::MAX;
                                            i_i_kernel = 0;
                                        } else {
                                            i_i_self = b_i * ns_i_self[0] + g_i * ns_i_self[1] + c_ii * ns_i_self[2] + (i_i_self_h - params.padding_h) * ns_i_self[3] + (i_i_self_w - params.padding_w);
                                            i_i_kernel = (c_i + g_i * params.channels_out) * ns_i_kernel[0] + c_ii * ns_i_kernel[1] + h_ii * ns_i_kernel[2] + w_ii;
                                        }

                                        i_self[i_i] = i_i_self;
                                        i_kernel[i_i] = i_i_kernel;
                                    }
                                }
                            }
                            i_res = b_i * ns_res[0] + g_i * ns_res[1] + c_i * ns_res[2] + h_i * ns_res[3] + w_i;
                            res[i_res] = i_self.iter().zip(i_kernel.iter()).map(| (i0, i1) | {
                                if *i0 == usize::MAX {
                                    return false;
                                }
                                self.data[*i0] * kernel.data[*i1]
                            }).sum::<bool>();
                        }
                    }
                }
            }
        }
        
        Self::from_vec(res)
    }
    
    
    fn flipped(&self, dims: &[usize], flip: &[usize]) -> WResult<Self> {
        
        let mut data = self.data.clone();
        let mut index_raw;
        let mut index_now;
        for flip_i in flip {
            let n0 = dims.iter().copied().take(*flip_i).reduce(| a, b | a * b).unwrap_or(1);
            let n = dims[*flip_i];
            let n1 = dims.iter().copied().skip(*flip_i+1).reduce(| a, b | a * b).unwrap_or(1);
            for n0_i in 0..n0 {
                for n1_i in 0..n1 {
                    for n_i in 0..(n / 2) {
                        index_raw = n0_i * n * n1 + n_i * n1 + n1_i;
                        index_now = n0_i * n * n1 + (n - n_i - 1) * n1 + n1_i;
                        data.swap(index_now, index_raw);
                    }
                }
            }

        }

        Self::from_vec(data)
    }



    /// 转置卷积
    /// 1. data元素间填充s-1行
    /// 2. data元素四周填充k-p-1行
    /// 3. kernel上下左右反转
    /// 4. 正常卷积：padding=0, stride=1
    // dim self: batch_size, groups * channels_in, data_h, data_w
    // dim kernel: channels_in, channels_out, data_h, data_w
    fn conv_transpose2d(&self, kernel: &Self, params: &ParamsConvTranspose2D) -> WResult<Self> {
        let (data_h, data_w, padding_h, padding_w) = params.get_data_hw()?;

        let data_len_prev = params.batch_size * params.groups * params.channels_in;
        let data_len_last = data_h * data_w;
        let data_len_last_raw = params.data_h * params.data_w;
        let data_len = data_len_prev * data_len_last;
        let mut data_now = vec![false; data_len];

        // data
        let mut now_i;
        let mut raw_i;
        let mut h_ii;
        let mut w_ii;
        for h_i in 0..params.data_h {
            for w_i in 0..params.data_w {
                h_ii = padding_h + params.stride_h * h_i;
                w_ii = padding_w + params.stride_w * w_i;
                for p_i in 0..data_len_prev {
                    now_i = h_ii * data_w + w_ii + p_i * data_len_last;
                    raw_i = h_i * params.data_w + w_i + p_i * data_len_last_raw;
                    // println!("p_i: {}, h_i: {}, w_i: {} -> {}", p_i, h_i, w_i, raw_i);
                    // println!("p_i: {}, h_i: {}/{}, w_i: {}/{} -> {}/{}", p_i, h_ii, data_h, w_ii, data_w, now_i, data_len);
                    data_now[now_i] = self.data[raw_i];
                }
            }
        }

        // kernel
        let kernel_now = kernel.transpose(
            &[params.channels_in, params.channels_out, params.kernel_h, params.kernel_w],
            0,
            1
        )?.flipped(
            &[params.channels_out, params.channels_in, params.kernel_h, params.kernel_w],
            &[2, 3]
        )?;

        // println!("kernel: {}, data: {}", kernel_now.len(), data_now.len());

        let params_conv2d = params.to_params_conv2d();
        
        Self::from_vec(data_now)?.conv2d(&kernel_now, &params_conv2d)
    }
}