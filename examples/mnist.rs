use kdam::{tqdm, BarExt};
use wnum::{array::error::WResult, dtype::cuda::{f32::{cuda_device_reset, cuda_device_set}, F32}, tensor::{base::WTensor, dataset::mnist::Mnist, log::log_init, module::{linear::Linear, loss::mse, optim::{Optim, SGD}}, varmap::Varmap}};



struct MyModel {
    pub lns: Vec<Linear<F32>>,
}

impl MyModel {
    fn new(dims: &[usize], varmap: &mut Varmap<F32>) -> Self {
        let lns = dims.windows(2).enumerate().map(| (i, v) | {
            Linear::new(v[0], v[1], true, &format!("ln_{}", i), varmap).unwrap()
        }).collect::<Vec<_>>();
        Self { lns }
    }

    fn forward(&self, data: WTensor<F32>) -> WResult<WTensor<F32>> {
        let mut y0 = data;
        let idx_last = self.lns.len() - 1;
        for (i, ln) in self.lns.iter().enumerate() {
            // log::debug!("y0 1: {}", y0);
            y0 = ln.forward(&y0)?;
            // log::debug!("y0 2: {}", y0);
            if i < idx_last {
                y0 = y0.relu()?;
                // log::debug!("y0 3: {}", y0);
            }
        }
        // log::debug!("y0 res: {}", y0);
        let res = y0.softmax(1)?;
        Ok(res)
    }
}



fn main() -> WResult<()> {
    log_init(true, false);
    cuda_device_reset();
    cuda_device_set(0);

    let dir_data = "/root/tasks/wnum/data/mnist";
    let batch_size = 1;
    let epoch = 32;
    let lr = 0.01;
    let dims = vec![784, 10];
    log::info!("model ...");
    let mut varmap = Varmap::default();
    let m = MyModel::new(&dims, &mut varmap);
    log::info!("data ...");
    let mut data = Mnist::<F32>::new(dir_data, true, batch_size)?;
    let batch_num = data.batch_num();
    let mut opt = SGD::new(lr);
    log::info!("start training ...");
    for epoch_idx in 0..epoch {
        let mut loss_all = 0.0;
        let mut pb = tqdm!(total = batch_num, desc = format!("Epoch {:2}/{}", epoch_idx+1, epoch));
        for (batch_idx, data_single) in data.by_ref().enumerate() {
        // for data_single in data.by_ref() {
            let logits = m.forward(data_single.data)?;
            let loss = mse(&logits, &data_single.label)?;
            let loss_scaler = loss.sum_all()?.to_scalar()? * 100.0;
            log::info!("epoch {}/{}, batch {}/{}, loss: {}", epoch_idx, epoch, batch_idx, batch_num, loss_scaler);
            if loss_scaler.is_nan() || loss_scaler.is_infinite() {
                break;
            }
            loss_all += loss_scaler;
            opt.backward(loss)?;
            pb.update(1)?;
            // pb.write(format!("epoch {}/{}, loss: {}", epoch_idx, epoch, loss_scaler))?;
            // if batch_idx > 4 {
            //     break;
            // }
        }
        pb.set_description(format!("Epoch {:2}/{}, loss: {:6.2}", epoch_idx+1, epoch, loss_all));
        pb.update_to(batch_num)?;
        println!();
        // println!("w: {} {}", epoch_idx, m.lns.last().unwrap().w);
        data.reset()?;
    }

    varmap.save("mnist.tensor")?;

    Ok(())
}