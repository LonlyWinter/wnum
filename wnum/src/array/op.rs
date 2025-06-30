
use std::vec::IntoIter;

use super::arr::WArr;
use super::dim::WArrDims;
use super::data::{Data, ParamsConv2D, ParamsConvTranspose2D};
use super::error::{WError, WResult};


macro_rules! data_methods {
    ($method:ident) => {
        pub fn $method(&self) -> WResult<Self> {
            let data = self.data.$method()?;
            Ok(Self::new(data, self.dims.clone()))
        }    
    };
    ($method:ident, "arr") => {
        pub fn $method(&self, rhs: &Self) -> WResult<Self> {
            if !self.dim_for_basic(rhs) {
                return Err(WError::ShapeNumMisMatch(self.dims.dims_num(), rhs.dims.dims_num()));
            }
            let data = self.data.$method(&rhs.data)?;
            Ok(Self::new(data, self.dims.clone()))
        }    
    };
    ($method:ident, "s") => {
        pub fn $method(&self, rhs: &T::Basic) -> WResult<Self> {
            let data = self.data.$method(rhs)?;
            Ok(Self::new(data, self.dims.clone()))
        }
    };
    ($method:ident, "dim") => {
        pub fn $method(&self, dim: u8) -> WResult<Self> {
            if dim >= self.dims.dims_num() {
                return Err(WError::DimNotFound(dim));
            }
            
            let mut dims = self.dims.to_vec();
            let dim_usize = dim as usize;
            let data = self.data.$method(&dims, dim_usize)?;
            dims[dim_usize] = 1;
            Ok(Self::new(data, WArrDims::from(dims.as_slice())))
        }
    };
    ($method:ident, $($name:ident: $ty:ty),*) => {
        pub fn $method(&self, $($name: $ty),*) -> WResult<Self> {
            let data = self.data.$method($($name),*)?;
            Ok(Self::new(data, self.dims.clone()))
        }    
    }
}


impl<T: Data> WArr<T> {
    fn dim_for_basic(&self, rhs: &Self) -> bool {
        if !self.dims.dims_same(&rhs.dims) {
            return false;
        }
        self.dims.to_vec().into_iter().zip(rhs.dims.to_vec()).all(| (a, b) | a == b)
    }

    pub fn reshape<F: Into<WArrDims>>(&self, shape: F) -> WResult<Self> {
        let dims = shape.into();
        let dims_inp = dims.dims_len();
        let dims_raw = self.data.len();
        if dims_inp != dims_raw {
            return Err(WError::DimMisMatch(dims_inp, dims_raw));
        }
        Ok(Self::new(self.data.clone(), dims))
    }

    pub fn unsqueeze(&self, dim: u8) -> WResult<Self> {
        let index = dim % (self.dims.dims_num() + 1);
        let dims = self.dims.dims_insert(index, 1);
        Ok(Self::new(self.data.clone(), dims))
    }

    pub fn squeeze(&self, dim: u8) -> WResult<Self> {
        let mut dims = self.dims.to_vec();
        let dim = (dim % self.dims.dims_num()) as usize;
        if dims[dim] != 1 {
            return Err(WError::DimMisMatch(1, dims[dim]));
        }
        dims.remove(dim);
        let dims = WArrDims::from(dims.as_slice());
        Ok(Self::new(self.data.clone(), dims))
    }
    

    pub fn from_vecs(data: Vec<Self>) -> WResult<Self> {
        let dims0 = data.first().unwrap();
        let dims0 = &dims0.dims;
        let dims = dims0.dims_insert(0, data.len());
        let fit = data.iter().all(| v | v.dims.eq(dims0));
        if !fit {
            panic!("dim not same");
        }
        let datas = data.into_iter().map(| v | v.data).collect::<Vec<_>>();
        let data = T::from_vec_data(datas)?;
        Ok(Self::new(data, dims))
    }
    
    pub fn to_vec(&self) -> WResult<Vec<T::Basic>> {
        self.data.to_vec()
    }

    pub fn zero(&self) -> T::Basic {
        self.data.zero()
    }

    pub fn one(&self) -> T::Basic {
        self.data.one()
    }

    pub fn neg_one(&self) -> T::Basic {
        self.data.neg_one()
    }

    // float
    data_methods!(sqrt);
    data_methods!(exp);
    data_methods!(log2);
    data_methods!(ln);
    data_methods!(round, n: T::Basic);
    // basic
    data_methods!(add, "arr");
    data_methods!(sub, "arr");
    data_methods!(div, "arr");
    data_methods!(mul, "arr");
    data_methods!(eq_item, "arr");
    data_methods!(neg);
    // broadcast
    data_methods!(broadcast_add, "s");
    data_methods!(broadcast_sub, "s");
    data_methods!(broadcast_div, "s");
    data_methods!(broadcast_mul, "s");
    // normal without dim
    data_methods!(abs);
    data_methods!(relu);
    
    pub fn sqr(&self) -> WResult<Self> {
        self.mul(self)
    }

    pub fn item_iter(&self) -> IntoIter<T::Basic> {
        self.data.item_iter()
    }

    pub fn map_item<F: Fn(&T::Basic) -> T::Basic>(&self, f: F) -> WResult<Self> {
        let data = self.data.map_item(f)?;
        Ok(Self::new(data, self.dims.clone()))
    }

    pub fn map_dim<F: Fn(IntoIter<T::Basic>) -> T::Basic>(&self, dim: u8, f: F) -> WResult<Self> {
        if dim >= self.dims.dims_num() {
            return Err(WError::DimNotFound(dim));
        }
        
        let mut dims = self.dims.to_vec();
        let dim_usize = dim as usize;
        let data = self.data.map_dim(&dims, dim_usize, f)?;
        dims[dim_usize] = 1;
        Ok(Self::new(data, WArrDims::from(dims.as_slice())))
    }
    
    // 数组乘法
    pub fn matmul(&self, rhs: &Self) -> WResult<Self> {
        let (data, dims) = match (&self.dims, &rhs.dims) {
            // 一维数组
            (WArrDims::Dim1(a), WArrDims::Dim1(b)) if a == b => {
                let res = self.data.matmul_1d(&rhs.data)?;
                (res, self.dims.clone())
            },
            // 二维数组
            (WArrDims::Dim2(a), WArrDims::Dim2(b)) if a.1 == b.0 => {
                let res = self.data.matmul_2d(
                    &rhs.data,
                    a.0,
                    b.1,
                    a.1
                )?;
                (res, WArrDims::Dim2((a.0, b.1)))
            },
            // 多维数组
            (a_raw, b_raw) if a_raw.dims_num() == b_raw.dims_num() && a_raw.dims_len() > 2 => {
                let mut a = a_raw.to_vec();
                let mut b = b_raw.to_vec();
                let a1 = a.pop().unwrap();
                let a0 = a.pop().unwrap();
                let b1 = b.pop().unwrap();
                let b0 = b.pop().unwrap();

                if a1 != b0 {
                    return Err(WError::DimMisMatch(a1, b0));
                }
                
                let mut all = 1;
                for (i, j) in a.iter().zip(b.iter()) {
                    if i == j {
                        all *= *i;
                        continue;
                    }
                    return Err(WError::DimMisMatch(*i, *j));
                }
                
                let res = self.data.matmul_nd(
                    &rhs.data,
                    a0,
                    b1,
                    a1,
                    all
                )?;
                
                a.push(a0);
                a.push(b1);
                let dim = WArrDims::from(a.as_slice());
                (res, dim)
            },
            // 其他数组
            (a, b) => {
                return Err(WError::ShapeNumMisMatch(
                    a.dims_num(),
                    b.dims_num()
                ));
            }
        };

        Ok(Self::new(data, dims))
    }




    data_methods!(mean, "dim");
    data_methods!(sum, "dim");
    data_methods!(max, "dim");
    data_methods!(min, "dim");
    data_methods!(where_cond, s: &T::Basic, a: &T::Basic, b: &T::Basic);


    pub fn transpose(&self, dim0: u8, dim1: u8) -> WResult<Self> {
        if dim0 >= self.dims.dims_num() {
            return Err(WError::DimNotFound(dim0));
        }
        if dim1 >= self.dims.dims_num() {
            return Err(WError::DimNotFound(dim1));
        }
        if dim0 == dim1 {
            return Ok(Self::new(self.data.clone(), self.dims.clone()));
        }
        let (dim0, dim1) = if dim0 > dim1 {
            (dim1, dim0)
        } else {
            (dim0, dim1)
        };

        let mut dims = self.dims.to_vec();
        let dim0 = dim0 as usize;
        let dim1 = dim1 as usize;

        let data = self.data.transpose(&dims, dim0, dim1)?;
        dims.swap(dim0, dim1);
        let dims = WArrDims::from(dims.as_slice());
        
        Ok(Self::new(data, dims))
    }

    pub fn t(&self) -> WResult<Self> {
        let n = self.dims.dims_num();
        if n < 2 {
            return Err(WError::DimNotFound(1));
        }
        let mut dims = self.dims.to_vec();
        let dim0 = n as usize - 2;
        let dim1 = n as usize - 1;
        
        let data = self.data.transpose(&dims, dim0, dim1)?;
        dims.swap(dim0, dim1);
        let dims = WArrDims::from(dims.as_slice());

        Ok(Self::new(data, dims))
    }

    pub fn concat(&self, rhs: &Self, dim: u8) -> WResult<Self> {
        if dim >= self.dims.dims_num() {
            return Err(WError::DimNotFound(dim));
        }
        let mut dims = self.dims.to_vec();
        let dims1 = rhs.dims.to_vec();
        let dim_usize = dim as usize;
        for (index, (i, j)) in dims.iter().zip(dims1.iter()).enumerate() {
            if i == j || index == dim_usize {
                continue;
            }
            return Err(WError::DimMisMatch(*i, *j));
        }
        
        dims[dim_usize] += dims1[dim_usize];

        let data = self.data.concat(&self.dims.to_vec(), &rhs.data, dim_usize, dims1[dim_usize])?;
        let dims = WArrDims::from(dims.as_slice());

        Ok(Self::new(data, dims))
    }

    pub fn stack(&self, rhs: &Self, dim: u8) -> WResult<Self> {
        let dim = dim % (self.dims.dims_num() + 1);
        let mut dims = self.dims.to_vec();
        let dim_usize = dim as usize;
        for (i, j) in dims.iter().zip(rhs.dims.to_vec().iter()) {
            if i == j {
                continue;
            }
            return Err(WError::DimMisMatch(*i, *j));
        }
        
        dims.insert(dim_usize, dim_usize);
        
        let data = self.data.stack(&self.dims.to_vec(), &rhs.data, dim_usize)?;
        let dims = WArrDims::from(dims.as_slice());
        
        Ok(Self::new(data, dims))
    }


    pub fn broadcast(&self, dim: u8, n: usize) -> WResult<Self> {
        if dim >= self.dims.dims_num() {
            return Err(WError::DimNotFound(dim));
        }

        let mut dims = self.dims.to_vec();
        let dim_now = dim as usize;
        let dim_usize = dims[dim_now];
        if n % dim_usize > 0 || dim_usize != 1 {
            return Err(WError::DimCanotRepeat(dim, dim_usize, n));
        }
        if n == dim_usize {
            return Ok(Self::new(self.data.clone(), self.dims.clone()));
        }

        dims[dim_now] = n;

        let data = self.data.broadcast(&self.dims.to_vec(), dim_now, n)?;
        let dims = WArrDims::from(dims.as_slice());

        Ok(Self::new(data, dims))
    }

    pub fn conv2d(&self, kernel: &Self, stride: usize, padding: usize, dilation: usize, groups: usize) -> WResult<Self> {
        self.conv2d_wh(
            kernel,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation,
            groups
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_wh(&self,
            kernel: &Self,
            stride_h: usize,
            stride_w: usize,
            padding_h: usize,
            padding_w: usize,
            dilation_w: usize,
            dilation_h: usize,
            groups: usize
        ) -> WResult<Self> {
        // w_dims[0] / groups = k
        // t_dims[1] / groups = w_dims[1]
        let self_dims = self.dims.to_vec();
        let w_dims = kernel.dims.to_vec();
        if w_dims[0] % groups != 0 {
            return Err(WError::DimNumError(format!("kernel dims0({}) % groups({}) != 0", w_dims[0], groups)));
        }
        if self_dims[1] / w_dims[1] != groups {
            return Err(WError::DimNumError(format!("kernel dims1({}) * groups({}) != data dims1({})", w_dims[1], groups, self_dims[1])));
        }

        let params = ParamsConv2D {
            batch_size: self_dims[0],
            data_w: self_dims[3],
            data_h: self_dims[2],
            kernel_w: w_dims[3],
            kernel_h: w_dims[2],
            padding_w,
            padding_h,
            stride_w,
            stride_h,
            dilation_w,
            dilation_h,
            groups,
            channels_out: w_dims[0] / groups,
            channels_in: w_dims[1],
        };

        let data = self.data.conv2d(&kernel.data, &params)?;
        let dims = params.get_output_dims()?;
        let dims = WArrDims::from(dims.as_slice());
        Ok(Self::new(data, dims))
    }


    
    pub fn conv_transpose2d(&self, kernel: &Self, stride: usize, padding: usize, output_padding: usize, dilation: usize, groups: usize) -> WResult<Self> {
        self.conv_transpose2d_wh(
            kernel,
            stride,
            stride,
            padding,
            padding,
            output_padding,
            output_padding,
            dilation,
            dilation,
            groups
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d_wh(&self,
            kernel: &Self,
            stride_h: usize,
            stride_w: usize,
            padding_h: usize,
            padding_w: usize,
            output_padding_w: usize,
            output_padding_h: usize,
            dilation_w: usize,
            dilation_h: usize,
            groups: usize
        ) -> WResult<Self> {
            let self_dims = self.dims.to_vec();
            let w_dims = kernel.dims.to_vec();
            if w_dims[1] % groups != 0 {
                return Err(WError::DimNumError(format!("kernel dims1({}) % groups({}) != 0", w_dims[0], groups)));
            }
            if self_dims[1] / w_dims[0] != groups {
                return Err(WError::DimNumError(format!("kernel dims0({}) * groups({}) != data dims1({})", w_dims[1], groups, self_dims[1])));
            }
            
            // dim self: batch_size, groups * channels_in, data_h, data_w
            // dim kernel: channels_in, channels_out, data_h, data_w
            let params = ParamsConvTranspose2D {
                batch_size: self_dims[0],
                data_w: self_dims[3],
                data_h: self_dims[2],
                kernel_w: w_dims[3],
                kernel_h: w_dims[2],
                padding_w,
                padding_h,
                output_padding_w,
                output_padding_h,
                stride_w,
                stride_h,
                dilation_w,
                dilation_h,
                groups,
                channels_out: w_dims[1],
                channels_in: w_dims[0],
            };

            let data = self.data.conv_transpose2d(&kernel.data, &params)?;
            let dims = params.get_output_dims()?;
            let dims = WArrDims::from(dims.as_slice());
            Ok(Self::new(data, dims))

        }
    
    pub fn gather(&self, indexes: &Self, dim: u8) -> WResult<Self> {
        if self.dims.dims_num() != indexes.dims.dims_num() {
            return Err(WError::DimMisMatch(0, 0))
        }
        let data = self.data.gather(
            &indexes.data,
            &self.dims.to_vec(),
            &indexes.dims.to_vec(),
            dim as usize
        )?;
        let dims = indexes.dims.clone();
        Ok(Self::new(data, dims))
    }

    pub fn scatter(&self, indexes: &Self, src: &Self, dim: u8) -> WResult<Self> {
        if src.dims.dims_num() != indexes.dims.dims_num() {
            return Err(WError::DimMisMatch(0, 0))
        }
        let data = self.data.scatter(
            &indexes.data,
            &src.data,
            &self.dims.to_vec(),
            &indexes.dims.to_vec(),
            dim as usize
        )?;
        let dims = self.dims.clone();
        Ok(Self::new(data, dims))
    }
}
