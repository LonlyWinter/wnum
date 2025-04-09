use std::{fs::File, io::Read, path::{Path, PathBuf}};

use flate2::read::GzDecoder;
use rand::{rng, seq::SliceRandom};

use crate::{array::{arr::WArr, data::Data, error::{WError, WResult}}, tensor::base::WTensor};



pub struct MnistBatch<T: Data> {
    pub data: WTensor<T>,
    pub label: WTensor<T>
}



pub struct Mnist<T: Data> {
    data: Vec<MnistBatch<T>>,
    meta: (PathBuf, PathBuf, usize),
}

impl<T: Data> Mnist<T> {
    fn check_file_exist(p: &Path, p2: &str) -> WResult<PathBuf> {
        let p = p.join(p2);
        if p.exists() && p.is_file() {
            return Ok(p);
        }
        Err(WError::Read(format!("file {:?} not exist", p)))
    }

    fn read_data(f: &Path, batch_size: usize) -> WResult<Vec<WTensor<T>>> {
        let f_data = File::open(f)?;
        let mut de_data = GzDecoder::new(f_data);
        let mut magic = [0u8; 4];
        de_data.read_exact(&mut magic)?;
        let mut n = [0u8; 4];
        de_data.read_exact(&mut n)?;
        let mut rows = [0u8; 4];
        de_data.read_exact(&mut rows)?;
        let mut cols = [0u8; 4];
        de_data.read_exact(&mut cols)?;
        let n = u32::from_be_bytes(n) as usize;
        let rows = u32::from_be_bytes(rows) as usize;
        let cols = u32::from_be_bytes(cols) as usize;
        let res_num = n / batch_size;
        let img_num = rows * cols * batch_size;
        (0..res_num).map(| _ |{
            let data = (0..img_num/4).flat_map(| _ |{
                let mut data_temp = [0u8; 4];
                let _ = de_data.read_exact(&mut data_temp);
                data_temp
            }).collect::<Vec<_>>();
            let data = WArr::<T>::from_vec1_f64(data)?.reshape((batch_size, rows*cols))?;
            WTensor::from_data(data, false)
        }).collect::<WResult<Vec<_>>>()
    }

    
    fn read_label(f: &Path, batch_size: usize) -> WResult<Vec<WTensor<T>>> {
        let f_data = File::open(f)?;
        let mut de_data = GzDecoder::new(f_data);

        let mut magic = [0u8; 4];
        de_data.read_exact(&mut magic)?;
        let mut n = [0u8; 4];
        de_data.read_exact(&mut n)?;
        let n = u32::from_be_bytes(n) as usize;
        let res_num = n / batch_size;
        (0..res_num).map(| _ |{
            let data = (0..batch_size).map(| _ | {
                let mut data_temp = [0u8; 1];
                let _ = de_data.read_exact(&mut data_temp);
                data_temp[0]
            })
            .flat_map(| v |{
                // one hot
                let mut res = [0u8; 10];
                let i = v as usize;
                res[i] = 1;
                res
            }).collect::<Vec<_>>();
            let data = WArr::<T>::from_vec1_f64(data)?.reshape((batch_size, 10))?;
            WTensor::from_data(data, false)
        }).collect::<WResult<Vec<_>>>()
    }


    pub fn new(dir_data: &str, train: bool, batch_size: usize) -> WResult<Self> {
        let p = PathBuf::from(dir_data);
        if !(p.exists() && p.is_dir()) {
            return Err(WError::Read(format!("dir {} not exist", dir_data)));
        }

        let (file_data, file_label) = if train {
            let p1 = Self::check_file_exist(&p, "train-images-idx3-ubyte.gz")?;
            let p2 = Self::check_file_exist(&p, "train-labels-idx1-ubyte.gz")?;
            (p1, p2)
        } else {
            let p1 = Self::check_file_exist(&p, "t10k-images-idx3-ubyte.gz")?;
            let p2 = Self::check_file_exist(&p, "t10k-labels-idx1-ubyte.gz")?;
            (p1, p2)
        };

        let mut res = Self {
            data: Vec::new(),
            meta: (file_data, file_label, batch_size)
        };
        res.reset()?;

        Ok(res)
    }

    pub fn batch_num(&self) -> usize {
        self.data.len()
    }

    pub fn reset(&mut self) -> WResult<()> {
        let data = Self::read_data(&self.meta.0, self.meta.2)?;
        let label = Self::read_label(&self.meta.1, self.meta.2)?;

        if data.len() != label.len() {
            return Err(WError::Read("data and label not match".to_string()));
        }

        let mut res = data.into_iter().zip(label).map(| (d, l) | {
            MnistBatch {
                data: d,
                label: l
            }
        }).collect::<Vec<_>>();

        let mut rng = rng();
        res.shuffle(&mut rng);

        #[cfg(feature = "logger")]
        log::debug!("MINST data num: {}", res.len());

        if res.is_empty() {
            return Err(WError::Read("No data".to_string()))
        }

        self.data = res;
        Ok(())
    }
}


impl<T: Data> Iterator for Mnist<T> {
    type Item = MnistBatch<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.data.pop()
    }
}


