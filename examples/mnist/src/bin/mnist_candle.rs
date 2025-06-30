use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, loss::cross_entropy, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use kdam::{tqdm, BarExt};
use wnum::{dtype::cpu::F32, tensor::dataset::mnist::Mnist as MnistData};



pub struct Mnist {
    lns: Vec<Linear>
}

impl Mnist {
    fn new(dims: &[usize], vb: VarBuilder) -> Result<Self> {
        let lns = dims.windows(2).enumerate().map(| (i, v) |{
            linear(v[0], v[1], vb.pp(format!("ln_{i}")))
        }).collect::<Result<Vec<Linear>>>()?;
        Ok(Self { lns })
    }

    fn forward(&self, data: Tensor) -> Result<Tensor> {
        let mut y0 = data;
        let idx_last = self.lns.len() - 1;
        for (i, ln) in self.lns.iter().enumerate() {
            y0 = ln.forward(&y0)?;
            if i < idx_last {
                y0 = y0.relu()?;
            }
        }
        Ok(y0)
    }
}

fn main() -> Result<()> {
    let dir_data = "/root/tasks/wnum_dev/data/mnist";
    let batch_size = 256;
    let epoch = 32;
    let lr = 0.001;
    let dims = vec![784, 128, 10];
    let dtype = DType::F32;
    let device = Device::Cpu;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let m = Mnist::new(&dims, vb)?;
    let mut data = MnistData::<F32>::new(dir_data, true, batch_size).unwrap();
    let mut opt = SGD::new(varmap.all_vars(), lr)?;
    let batch_num = data.batch_num();

    for epoch_idx in 0..epoch {
        let mut loss_all = 0.0;
        let mut pb = tqdm!(
            total = batch_num,
            desc = format!("Epoch {:2}/{}", epoch_idx+1, epoch)
        );
        for data_single in data.by_ref() {
            let data_inp = Tensor::from_vec(
                data_single.data.to_vec().unwrap(),
                (batch_size, 784),
                &device
            )?.to_dtype(dtype)?;
            let data_opt = Tensor::from_vec(
                data_single.label.to_vec().unwrap(),
                (batch_size, ),
                &device
            )?.to_dtype(DType::U8)?;
            let logits = m.forward(data_inp)?;
            let loss = cross_entropy(&logits, &data_opt)?;
            let loss_scaler = loss.to_scalar::<f32>()?;
            if loss_scaler.is_nan() || loss_scaler.is_infinite() {
                break;
            }
            loss_all += loss_scaler;
            opt.backward_step(&loss)?;
            pb.update(1)?;
        }
        pb.set_description(format!("Epoch {:2}/{}, loss: {:6.2}", epoch_idx+1, epoch, loss_all));
        pb.update_to(batch_num)?;
        println!();
        data.reset().unwrap();
    }

    Ok(())
}