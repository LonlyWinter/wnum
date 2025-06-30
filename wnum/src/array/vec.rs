
use super::{arr::WArr, data::Data, dim::WArrDims, error::WResult};



// impl<T> From<Vec<Self>> for WArr<T>
// where
//     T: Data
// {
//     fn from(value: Vec<Self>) -> Self {
//         Self::from_vec_vec(value).unwrap()
//     }
// }


macro_rules! warr_from_vec {
    ($from_name0:ident, $from_name1:ident, $from_name2:ident, $from_name3:ident, $ty0:ty, $ty1:ty) => {
        #[allow(clippy::type_complexity)]
        pub fn $from_name1(data: $ty0) -> WResult<WArr<T>> {
            let mut data_now = Vec::with_capacity(data.len());
            for data_single in data.into_iter() {
                let data_temp = Self::$from_name0(data_single)?;
                data_now.push(data_temp);
            }
            WArr::from_vecs(data_now)
        }

        #[allow(clippy::type_complexity)]
        pub fn $from_name2<V: Into<f64>>(data: $ty1) -> WResult<WArr<T>> {
            let mut data_now = Vec::with_capacity(data.len());
            for data_single in data.into_iter() {
                let data_temp = Self::$from_name3(data_single)?;
                data_now.push(data_temp);
            }
            WArr::from_vecs(data_now)
        }
    }
}

impl<T: Data> WArr<T> {
    pub fn from_vec1(data: Vec<T::Basic>) -> WResult<WArr<T>> {
        let dims = WArrDims::Dim1(data.len());
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    pub fn from_vec1_f64<V: Into<f64>>(data: Vec<V>) -> WResult<WArr<T>> {
        let dims = WArrDims::Dim1(data.len());
        let data = data
            .into_iter()
            .map(| v | T::f64_to_basic(v.into()))
            .collect::<WResult<Vec<_>>>()?;
        let data = T::from_vec(data)?;
        Ok(WArr::new(data, dims))
    }

    warr_from_vec!(from_vec1, from_vec2, from_vec2_f64, from_vec1_f64, Vec<Vec<T::Basic>>, Vec<Vec<V>>);
    warr_from_vec!(from_vec2, from_vec3, from_vec3_f64, from_vec2_f64, Vec<Vec<Vec<T::Basic>>>, Vec<Vec<Vec<V>>>);
    warr_from_vec!(from_vec3, from_vec4, from_vec4_f64, from_vec3_f64, Vec<Vec<Vec<Vec<T::Basic>>>>, Vec<Vec<Vec<Vec<V>>>>);
    warr_from_vec!(from_vec4, from_vec5, from_vec5_f64, from_vec4_f64, Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>, Vec<Vec<Vec<Vec<Vec<V>>>>>);
    warr_from_vec!(from_vec5, from_vec6, from_vec6_f64, from_vec5_f64, Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>);
    warr_from_vec!(from_vec6, from_vec7, from_vec7_f64, from_vec6_f64, Vec<Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>>);
    warr_from_vec!(from_vec7, from_vec8, from_vec8_f64, from_vec7_f64, Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<T::Basic>>>>>>>>, Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<V>>>>>>>>);
}


