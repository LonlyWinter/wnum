use kdam::{tqdm, BarExt};
use wnum::{array::error::WResult, dtype::cpu::F32, tensor::{base::WTensor, dataset::mnist::Mnist, log::log_init, module::{linear::Linear, loss::cross_entropy, optim::{Optim, SGD}}, varmap::Varmap}};

struct MyModel {
    pub lns: Vec<Linear<F32>>,
}

impl MyModel {
    fn new(dims: &[usize], varmap: &mut Varmap<F32>) -> Self {
        let lns = dims.windows(2).enumerate().map(| (i, v) | {
            Linear::new(v[0], v[1], true, &format!("ln_{i}"), varmap).unwrap()
        }).collect::<Vec<_>>();
        Self { lns }
    }

    fn forward(&self, data: WTensor<F32>) -> WResult<WTensor<F32>> {
        let mut y0 = data;
        let idx_last = self.lns.len() - 1;
        for (i, ln) in self.lns.iter().enumerate() {
            // log::debug!("i {i}, y0 before: {:?}", y0);
            y0 = ln.forward(&y0)?;
            // log::debug!("i {i}, y0 after: {:?}", y0);
            if i < idx_last {
                y0 = y0.relu()?;
            }
        }
        Ok(y0)
    }
}



fn main() -> WResult<()> {
    log_init(false, true);

    let dir_data = "/root/tasks/wnum_dev/data/mnist";
    let batch_size = 256;
    let epoch = 32;
    let lr = 0.001;
    let dims = vec![784, 128, 10];
    log::info!("model ...");
    let mut varmap = Varmap::default();
    let m = MyModel::new(&dims, &mut varmap);
    log::info!("data ...");
    let mut data = Mnist::<F32>::new(dir_data, true, batch_size)?;
    let batch_num = data.batch_num();
    let mut opt = SGD::new(lr);
    log::debug!("{:?}", varmap);
    log::info!("start training ...");
    for epoch_idx in 0..epoch {
        let mut loss_all = 0.0;
        let mut pb = tqdm!(total = batch_num, desc = format!("Epoch {:2}/{}", epoch_idx+1, epoch));
        // for (batch_idx, data_single) in data.by_ref().enumerate() {
        for data_single in data.by_ref() {
            let logits = m.forward(data_single.data)?;
            let loss = cross_entropy(&logits, &data_single.label)?;
            let loss_scaler = loss.to_vec()?.into_iter().sum::<f32>();
            if loss_scaler.is_nan() || loss_scaler.is_infinite() {
                break;
            }
            loss_all += loss_scaler;
            opt.backward(loss)?;
            pb.update(1)?;
            // pb.write(format!("Epoch {epoch_idx}/{epoch}, loss: {loss_scaler}"))?;
            // if batch_idx > 4 {
            //     break;
            // }
        }
        pb.set_description(format!("Epoch {:2}/{} {}/{}, loss: {:6.2}", epoch_idx+1, epoch, pb.counter, batch_num, loss_all));
        pb.refresh()?;
        println!();
        data.reset()?;
        // break;
    }
    // varmap.save("mnist.tensor")?;

    Ok(())
}
