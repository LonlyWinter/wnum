

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WArrDims {
    Dim1(usize),
    Dim2((usize, usize)),
    Dim3((usize, usize, usize)),
    Dim4((usize, usize, usize, usize)),
    Dim5((usize, usize, usize, usize, usize)),
    Dim6((usize, usize, usize, usize, usize, usize)),
    Dim7((usize, usize, usize, usize, usize, usize, usize)),
    Dim8((usize, usize, usize, usize, usize, usize, usize, usize)),
}




impl WArrDims {
    pub fn dims_same(&self, rhs: &Self) -> bool {
        matches!((self, rhs), (Self::Dim1(_), Self::Dim1(_)) | (Self::Dim2(_), Self::Dim2(_)) | (Self::Dim3(_), Self::Dim3(_)) | (Self::Dim4(_), Self::Dim4(_)) | (Self::Dim5(_), Self::Dim5(_)) | (Self::Dim6(_), Self::Dim6(_)) | (Self::Dim7(_), Self::Dim7(_)) | (Self::Dim8(_), Self::Dim8(_)))
    }

    pub fn dims_num(&self) -> u8 {
        match self {
            Self::Dim1(_) => 1,
            Self::Dim2(_) => 2,
            Self::Dim3(_) => 3,
            Self::Dim4(_) => 4,
            Self::Dim5(_) => 5,
            Self::Dim6(_) => 6,
            Self::Dim7(_) => 7,
            Self::Dim8(_) => 8,
        }
    }

    fn from_slice(dims: &[usize]) -> Self {
        match *dims {
            [a] => Self::Dim1(a),
            [a, b] => Self::Dim2((a, b)),
            [a, b, c] => Self::Dim3((a, b, c)),
            [a, b, c, d] => Self::Dim4((a, b, c, d)),
            [a, b, c, d, e] => Self::Dim5((a, b, c, d, e)),
            [a, b, c, d, e, f] => Self::Dim6((a, b, c, d, e, f)),
            [a, b, c, d, e, f, g] => Self::Dim7((a, b, c, d, e, f, g)),
            [a, b, c, d, e, f, g, h] => Self::Dim8((a, b, c, d, e, f, g, h)),
            _ => {
                panic!("dim too much");
            }
        }
    }

    pub fn dims_insert(&self, index: u8, n: usize) -> Self {
        let mut dims = self.to_vec();
        dims.insert(index as usize, n);
        Self::from_slice(&dims)
    }


    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Dim1(a) => vec![*a],
            Self::Dim2((a, b)) => vec![*a, *b],
            Self::Dim3((a, b, c)) => vec![*a, *b, *c],
            Self::Dim4((a, b, c, d)) => vec![*a, *b, *c, *d],
            Self::Dim5((a, b, c, d, e)) => vec![*a, *b, *c, *d, *e],
            Self::Dim6((a, b, c, d, e, f)) => vec![*a, *b, *c, *d, *e, *f],
            Self::Dim7((a, b, c, d, e, f, g)) => vec![*a, *b, *c, *d, *e, *f, *g],
            Self::Dim8((a, b, c, d, e, f, g, h)) => vec![*a, *b, *c, *d, *e, *f, *g, *h],
        }
    }

    pub fn dims_len(&self) -> usize {
        self.to_vec().into_iter().reduce(| a, b | a * b).unwrap_or(0)
    }

    pub fn dim_swap(&mut self, dim0: usize, dim1: usize) {
        let mut dims = self.to_vec();
        dims.swap(dim0, dim1);
        *self = Self::from_slice(&dims);
    }
}


macro_rules! warr_dim_from {
    ($ty:ty, $name:ident) => {
        impl From<$ty> for WArrDims {
            fn from(value: $ty) -> Self {
                let res = Self::$name(value);
                let dims = res.to_vec();
                if dims.into_iter().any(| v | v == 0) {
                    panic!("Dim contains zero: {:?}", value);
                }
                res
            }
        }  
    };
}

warr_dim_from!(usize, Dim1);
warr_dim_from!((usize, usize), Dim2);
warr_dim_from!((usize, usize, usize), Dim3);
warr_dim_from!((usize, usize, usize, usize), Dim4);
warr_dim_from!((usize, usize, usize, usize, usize), Dim5);
warr_dim_from!((usize, usize, usize, usize, usize, usize), Dim6);
warr_dim_from!((usize, usize, usize, usize, usize, usize, usize), Dim7);
warr_dim_from!((usize, usize, usize, usize, usize, usize, usize, usize), Dim8);


impl From<&[usize]> for WArrDims {
    fn from(value: &[usize]) -> Self {
        Self::from_slice(value)
    }
}
