
use std::fs::remove_file;

use wnum::array::arr::WArr;
use wnum::array::error::WResult;
use wnum::dtype::cpu::F32;
use wnum::tensor::base::WTensor;
use wnum::tensor::module::loss::cross_entropy;
use wnum::tensor::module::loss::nll;
use wnum::tensor::module::norm::Norm;
use wnum::tensor::{module::{linear::Linear, loss::mse, optim::{Optim, SGD}}, varmap::Varmap};



#[test]
fn sgd_linear_regression() -> WResult<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen: WTensor<F32> = WTensor::from_vec2(vec![vec![3f32], vec![1.]])?;
    let b_gen = WTensor::from_vec2(vec![vec![-2f32]])?;
    let ys_gen = WTensor::from_vec2(vec![vec![5.0f32], vec![23.0], vec![-2.0], vec![21.0]])?;
    let g = Linear::new_with_tensor(w_gen, Some(b_gen))?;
    let sample_xs = WTensor::from_vec2(vec![vec![2f32, 1.], vec![7., 4.], vec![-4., 12.], vec![5., 8.]])?;
    let sample_ys = g.forward(&sample_xs)?;
    let sample_ys_data = sample_ys.read_data()?.clone();

    assert_eq!(sample_ys, ys_gen, "ys");

    let ys_log = [
        WTensor::from_vec2(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]])?,
        WTensor::from_vec2(vec![vec![1.712], vec![5.998], vec![3.6060004], vec![6.79]])?,
    ];
    let w_log = [
        WTensor::from_vec2(vec![vec![0.568], vec![0.48200002]])?,
        WTensor::from_vec2(vec![vec![1.0061281], vec![0.71740806]])?,
    ];
    let b_log = [
        WTensor::from_vec2(vec![vec![0.094000004]])?,
        WTensor::from_vec2(vec![vec![0.151788]])?,
    ];
    let loss0_log = [
        WTensor::from_vec2(vec![vec![25.0], vec![529.0], vec![4.0], vec![441.0]])?,
        WTensor::from_vec2(vec![vec![10.810945], vec![289.06796], vec![31.42724], vec![201.9241]])?,
    ];
    let loss_log = [
        WTensor::from_vec2(vec![vec![999.0]])?,
        WTensor::from_vec2(vec![vec![533.2302]])?,
    ];
    let w_final = WTensor::from_vec2(vec![vec![2.4595], vec![0.8234]])?;
    let b_final = WTensor::from_vec2(vec![vec![0.2609]])?;

    let mut sgd = SGD::new(0.001);
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w_data = WArr::from_vec2(vec![vec![0f32], vec![0.0]])?;
    let w = WTensor::from_data(w_data, true)?;
    let b_data = WArr::from_vec2(vec![vec![0f32]])?;
    let b = WTensor::from_data(b_data, true)?;
    let lin = Linear::new_with_tensor(w.clone(), Some(b.clone()))?;
    for step in 0..2 {
        let ys = lin.forward(&sample_xs)?;
        assert_eq!(ys, ys_log[step], "ys {step}");
        
        let loss0 = ys.sub(&sample_ys)?.sqr()?;
        assert_eq!(loss0, loss0_log[step], "loss0 {step}");

        let loss = loss0.sum_all()?;
        assert_eq!(loss, loss_log[step], "loss {step}");
        
        sgd.backward(loss)?;
        assert_eq!(w, w_log[step], "w {step}");
        assert_eq!(b, b_log[step], "b {step}");
    }
    for _step in 0..8 {
        let sample_xs = WTensor::from_vec2(vec![vec![2f32, 1.], vec![7., 4.], vec![-4., 12.], vec![5., 8.]])?;
        let sample_ys = WTensor::from_data(sample_ys_data.clone(), false)?;

        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        sgd.backward(loss)?;
    }
    w.round(4.0)?;
    b.round(4.0)?;
    assert_eq!(w, w_final, "w final");
    assert_eq!(b, b_final, "b final");
    Ok(())
}






#[test]
fn sgd_linear_regression2() -> WResult<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen: WTensor<F32> = WTensor::from_vec2(vec![vec![3f32], vec![1.]])?;
    let b_gen = WTensor::from_vec2(vec![vec![-2f32]])?;
    let ys_gen = WTensor::from_vec2(vec![vec![5.0f32], vec![23.0], vec![-2.0], vec![21.0]])?;
    let g = Linear::new_with_tensor(w_gen, Some(b_gen))?;
    let sample_xs = WTensor::from_vec2(vec![vec![2f32, 1.], vec![7., 4.], vec![-4., 12.], vec![5., 8.]])?;
    let sample_ys = g.forward(&sample_xs)?;
    let sample_ys_data = sample_ys.read_data()?.clone();

    assert_eq!(sample_ys, ys_gen, "ys");

    let ys_log = [
        WTensor::from_vec2(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]])?,
        WTensor::from_vec2(vec![vec![0.428], vec![1.4995], vec![0.9015001], vec![1.6975]])?,
    ];
    let w_log = [
        WTensor::from_vec2(vec![vec![0.142], vec![0.120500006]])?,
        WTensor::from_vec2(vec![vec![0.27588302], vec![0.22558801]])?,
    ];
    let b_log = [
        WTensor::from_vec2(vec![vec![0.023500001]])?,
        WTensor::from_vec2(vec![vec![0.04473675]])?,
    ];
    let loss_log = [
        WTensor::from_vec2(vec![vec![249.75]])?,
        WTensor::from_vec2(vec![vec![216.04497]])?,
    ];
    let w_final = WTensor::from_vec2(vec![vec![1.1132], vec![0.6938]])?;
    let b_final = WTensor::from_vec2(vec![vec![0.1568]])?;

    let mut sgd = SGD::new(0.001);
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    
    let w_data = WArr::from_vec2(vec![vec![0f32], vec![0.0]])?;
    let w = WTensor::from_data(w_data, true)?;
    let b_data = WArr::from_vec2(vec![vec![0f32]])?;
    let b = WTensor::from_data(b_data, true)?;
    let lin = Linear::new_with_tensor(w.clone(), Some(b.clone()))?;
    for step in 0..2 {
        let ys = lin.forward(&sample_xs)?;
        assert_eq!(ys, ys_log[step], "ys {step}");
        
        let loss = mse(&ys, &sample_ys)?;
        assert_eq!(loss, loss_log[step], "loss {step}");
        
        sgd.backward(loss)?;
        assert_eq!(w, w_log[step], "w {step}");
        assert_eq!(b, b_log[step], "b {step}");
    }
    for _step in 0..8 {
        let sample_xs = WTensor::from_vec2(vec![vec![2f32, 1.], vec![7., 4.], vec![-4., 12.], vec![5., 8.]])?;
        let sample_ys = WTensor::from_data(sample_ys_data.clone(), false)?;

        let ys = lin.forward(&sample_xs)?;
        let loss = mse(&ys, &sample_ys)?;
        sgd.backward(loss)?;
    }
    w.round(4.0)?;
    b.round(4.0)?;
    assert_eq!(w, w_final, "w final");
    assert_eq!(b, b_final, "b final");
    Ok(())
}


#[test]
fn test_varmap() -> WResult<()> {
    let p = "data.varmap.log";
    let mut varmap = Varmap::<F32>::default();
    let a_data = WArr::random_normal((4, 3))?;
    let b_data = WArr::random_normal((3, 4))?;

    let a_bak = WTensor::from_data(a_data.clone(), false)?;
    let b_bak = WTensor::from_data(b_data.clone(), false)?;

    let a = WTensor::from_data(a_data, false)?;
    let b = WTensor::from_data(b_data, false)?;
    varmap.add_tensor("name_a", &a);
    varmap.add_tensor("name_b", &b);

    varmap.save(p)?;

    let a_zeros = a.zeros_like()?;
    let a_zeros_data = a_zeros.read_data()?.clone();
    let b_zeros = b.zeros_like()?;
    let b_zeros_data = b_zeros.read_data()?.clone();

    a.write_data(a_zeros_data)?;
    b.write_data(b_zeros_data)?;
    assert_eq!(a, a_zeros, "a set zero");
    assert_eq!(b, b_zeros, "b set zero");
    
    varmap.load(p)?;
    assert_eq!(a, a_bak, "a load");
    assert_eq!(b, b_bak, "b load");

    remove_file(p)?;
    Ok(())
}



#[test]
fn test_nll_and_cross_entropy() -> WResult<()> {
    let a: WTensor<F32> = WTensor::from_vec2(vec![
        vec![1.1050f32, 0.3013, -1.5394, -2.1528, -0.8634],
        vec![1.0730, -0.9419, -0.1670, -0.6582, 0.5061],
        vec![0.8318, 1.1154, -0.3610, 0.5351, 1.0830],
    ])?;
    let b = WTensor::from_vec1(vec![1.0, 0.0, 4.0])?;
    let res1_real = WTensor::from_vec2(vec![vec![-0.52879, -1.33249, -3.17319, -3.78659, -2.49719], vec![-0.77338, -2.78828, -2.01338, -2.50458, -1.34028], vec![-1.53904, -1.25544, -2.73184, -1.83574, -1.28784]])?;
    let res23_real = WTensor::from_vec2(vec![vec![1.1312]])?;
    
    let res1 = a.log_softmax(1)?;
    res1.round(5.0)?;
    assert_eq!(res1, res1_real, "res1");

    let res2 = nll(&res1, &b)?;
    res2.round(4.0)?;
    assert_eq!(res2, res23_real, "res2");

    let res3 = cross_entropy(&a, &b)?;
    res3.round(4.0)?;
    assert_eq!(res3, res23_real, "res3");

    Ok(())
}



#[test]
fn test_grad_gather_3() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![1.0, 0.0, 2.0]])?;
    let b = WTensor::from_data(b_data_real.clone(), false)?;
    let c_data_real = WArr::from_vec1_f64(vec![0.0, 1.0])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    
    let id_a = a.id();
    let y0 = a.gather(&b, 1)?;
    let y1 = nll(&y0, &c)?;
    let mut grads = y1.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let grad_a_real = WArr::from_vec2_f64(vec![vec![-0.5, 0.0, 0.0], vec![-0.5, 0.0, 0.0]])?;
    let y1_real = WArr::from_vec2_f64(vec![vec![-1.5]])?;
    let y0_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![4.0, 3.0, 5.0]])?;
    let y1_data = y1.read_data()?.clone();
    let y0_data = y0.read_data()?.clone();
    
    assert_eq!(y1_data, y1_real, "y1 data");
    assert_eq!(y0_data, y0_real, "y0 data");
    assert_eq!(grad_a, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_unsqueeze_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])?;
    let c_data_real = WArr::from_vec1_f64(vec![3])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    let grad_a_real = WArr::from_vec1_f64(vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-3.0]])?;
    
    let id_a = a.id();
    let b = a.unsqueeze(0)?;
    let loss = nll(&b, &c)?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    
    assert_eq!(loss_data, loss_data_real, "loss data");
    assert_eq!(b_data, b_data_real, "b data");
    assert_eq!(grad_a, grad_a_real, "grad a");
    Ok(())
}


#[test]
fn test_grad_logsofmax_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![-5.4562, -4.4562, -3.4562, -2.4562, -1.4562, -0.4562]])?;
    let c_data_real = WArr::from_vec1_f64(vec![3])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    let grad_a_real = WArr::from_vec1_f64(vec![0.0043, 0.0116, 0.0315, -0.9142, 0.2331, 0.6337])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![2.4562]])?;
    
    let id_a = a.id();
    let b = a.unsqueeze(0)?.log_softmax(1)?;
    let loss = nll(&b, &c)?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    assert_eq!(loss_data.round(4.0)?, loss_data_real, "loss data");
    assert_eq!(b_data.round(4.0)?, b_data_real, "b data");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_unsqueeze_logsofmax_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![-5.4562, -4.4562, -3.4562, -2.4562, -1.4562, -0.4562]])?;
    let c_data_real = WArr::from_vec1_f64(vec![3])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    let grad_a_real = WArr::from_vec1_f64(vec![0.0043, 0.0116, 0.0315, -0.9142, 0.2331, 0.6337])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![2.4562]])?;
    
    let id_a = a.id();
    let b = a.unsqueeze(0)?.log_softmax(1)?;
    let loss = nll(&b, &c)?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    assert_eq!(loss_data.round(4.0)?, loss_data_real, "loss data");
    assert_eq!(b_data.round(4.0)?, b_data_real, "b data");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}

#[test]
fn test_grad_unsqueeze_sofmax_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]])?;
    let c_data_real = WArr::from_vec1_f64(vec![3])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    let grad_a_real = WArr::from_vec1_f64(vec![0.0004, 0.0010, 0.0027, -0.0784, 0.0200, 0.0543])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-0.0858]])?;
    
    let id_a = a.id();
    let b = a.unsqueeze(0)?.softmax(1)?;
    let loss = nll(&b, &c)?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    assert_eq!(loss_data.round(4.0)?, loss_data_real, "loss data");
    assert_eq!(b_data.round(4.0)?, b_data_real, "b data");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}

// https://github.com/huggingface/candle/blob/main/candle-nn/tests/layer_norm.rs
#[test]
fn test_norm_1() -> WResult<()> {
    let w: WTensor<F32> = WTensor::from_vec1_f64(vec![3f32])?;
    let b: WTensor<F32> = WTensor::from_vec1_f64(vec![0.5f32])?;
    let ln2 = Norm::new_with_data(
        WTensor::cat(vec![&w, &w], 0)?,
        Some(WTensor::cat(vec![&b, &b], 0)?),
        1e-8,
        true
    )?;
    let ln3 = Norm::new_with_data(
        WTensor::cat(vec![&w, &w, &w], 0)?,
        Some(WTensor::cat(vec![&b, &b, &b], 0)?),
        1e-8,
        true
    )?;
    let ln = Norm::new_with_data(w, Some(b), 1e-8, true)?;

    println!("0");

    let two = WTensor::from_vec3_f64(vec![vec![vec![2f32]]])?;
    let res = ln.forward(&two)?;
    let res_real = WTensor::from_vec3_f64(vec![vec![vec![0.5f32]]])?;
    assert_eq!(res, res_real, "a");

    println!("a");

    let inp = WTensor::from_vec3_f64(vec![vec![vec![4f32, 0f32]]])?;
    let res = ln2.forward(&inp)?;
    let res_real = WTensor::from_vec3_f64(vec![vec![vec![3.5f32, -2.5]]])?;
    assert_eq!(res, res_real, "b");

    println!("b");

    let inp = WTensor::from_vec3_f64(vec![vec![
        vec![1f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![9.0, 8.0, 7.0],
    ]])?;
    let res = ln3.forward(&inp)?;
    let res_real = WTensor::from_vec3_f64(vec![vec![
        vec![-3.1742, 0.5, 4.1742],
        vec![-3.1742, 0.5, 4.1742],
        vec![4.1742, 0.5, -3.1742]
    ]])?;
    assert_eq!(res, res_real, "c");
    
    let mean = res.sum_keepdim(2)?.broadcast_div(&3.0)?;
    let mean_real = WTensor::from_vec3_f64(vec![vec![
        vec![0.5],
        vec![0.5],
        vec![0.5],
    ]])?;
    assert_eq!(mean, mean_real, "mean");

    let std = res.sub(&mean)?
        .sqr()?
        .sum_keepdim(2)?
        .sqrt()?
        .broadcast_div(&3.0)?;
    let std_real = WTensor::from_vec3_f64(vec![vec![
        vec![1.7321],
        vec![1.7321],
        vec![1.7321],
    ]])?;
    assert_eq!(std, std_real, "std");

    Ok(())
}