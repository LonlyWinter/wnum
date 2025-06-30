use std::{collections::HashMap, fmt::Debug, fs::File, io::{Read, Write}, os::unix::fs::FileExt, path::PathBuf};


use serde::{Deserialize, Serialize};

use crate::array::{arr::WArr, data::Data, dim::WArrDims, error::{WError, WResult}};

use super::base::WTensor;




pub struct Varmap<T: Data>(HashMap<String, WTensor<T>>);


impl<T: Data> Default for Varmap<T> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}


impl From<serde_json::Error> for WError {
    fn from(value: serde_json::Error) -> Self {
        Self::Read(format!("Serde: {value}"))
    }
}


#[derive(Debug, Serialize, Deserialize)]
struct FileMeta {
    name: Vec<String>,
    dims: HashMap<String, Vec<usize>>
}



impl<T: Data> Varmap<T> {
    pub fn add_tensor(&mut self, name: &str, data: &WTensor<T>) {
        self.0.insert(name.to_string(), data.clone());
    }

    pub fn gen_tensor_random_normal<S: Into<WArrDims>>(&mut self, name: &str, shape: S) -> WResult<WTensor<T>> {
        let res = WTensor::random_normal(shape)?;
        self.add_tensor(name, &res);
        Ok(res)
    }

    pub fn gen_tensor_random_uniform<S: Into<WArrDims>>(&mut self, name: &str, shape: S) -> WResult<WTensor<T>> {
        let res = WTensor::random(shape)?;
        self.add_tensor(name, &res);
        self.0.insert(name.to_string(), res.clone());
        Ok(res)
    }

    pub fn save(&self, file_data: &str) -> WResult<()> {
        let p = PathBuf::from(file_data);
        let mut name = Vec::new();
        let mut dims = HashMap::new();
        let mut data_all = Vec::new();
        
        for (k, v) in self.0.iter() {
            let data_temp = v.read_data()?.clone();
            name.push(k.to_owned());
            dims.insert(k.to_owned(), data_temp.dims.to_vec());
            data_all.extend(data_temp.item_iter());
        }
        let meta = serde_json::to_vec(&FileMeta { name: name.clone(), dims })?;
        let meta_size = (meta.len() as u64).to_le_bytes();
        let data_size = (data_all.len() as u64).to_le_bytes();
        let len_data = data_all.len() * std::mem::size_of::<T>();
        let data = unsafe {
            std::slice::from_raw_parts(data_all.as_ptr() as *const u8, len_data)
        };
        
        let mut f = File::create(p)?;
        f.write_all(&meta_size)?;
        f.write_all(&data_size)?;
        f.write_all(&meta)?;
        f.write_all(data)?;

        Ok(())
    }

    pub fn load(&mut self, file_data: &str) -> WResult<()> {
        let p = PathBuf::from(file_data);
        let mut f = File::open(p)?;
        
        let mut len_meta = [0u8; 8];
        f.read_exact(&mut len_meta)?;
        let len_meta = u64::from_le_bytes(len_meta) as usize;
        
        let mut data_size = [0u8; 8];
        f.read_exact(&mut data_size)?;
        let data_size = u64::from_le_bytes(data_size) as usize;
        
        let mut meta = vec![0u8; len_meta];
        f.read_exact_at(&mut meta, 16)?;
        let mut meta = serde_json::from_slice::<FileMeta>(&meta)?;
        
        let mut data = vec![0u8; data_size * std::mem::size_of::<T::Basic>()];
        f.read_exact_at(&mut data, 16 + len_meta as u64)?;
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const T::Basic, data_size)
        };
        
        let mut prev = 0;
        for name in meta.name.into_iter() {
            let dims = meta.dims.remove(&name).unwrap();
            let dims = WArrDims::from(dims.as_slice());
            let len_data = dims.dims_len();
            let last = prev + len_data;
            let data_temp = data[prev..last].to_vec();
            prev = last;
            let data_temp = T::from_vec(data_temp)?;
            let data_arr = WArr::new(data_temp, dims);
            if let Some(res) = self.0.get(&name) {
                res.write_data(data_arr)?;
            }
        }
        Ok(())
    }
}

impl<T: Data> Debug for Varmap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vars[{}]", self.0.iter().map(| v | format!("{} => {}", v.0, v.1.id())).collect::<Vec<_>>().join(", "))
    }
}