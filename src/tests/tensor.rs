
use std::fs::remove_file;

use crate::{dtype::cpu::{F32, U32}, tensor::{module::{linear::Linear, loss::mse, optim::{Optim, SGD}}, varmap::Varmap}};
#[allow(unused)]
use crate::{array::{arr::WArr, error::WResult}, tensor::{base::WTensor, log::log_init}};


#[test]
fn test_grad1() -> WResult<()> {
    // log_init(false, false);
    
    let x_data_real: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let y_data_real = WArr::from_vec1(vec![28f32, 10., 40.])?;
    let x = WTensor::from_data(x_data_real.clone(), true)?;
    let id_x = x.id();
    let y = x.mul(&x)?.add(&x.broadcast_mul(&5.0)?)?.broadcast_add(&4.0)?;
    let mut grads = y.backward()?;
    
    let grad_x = grads.remove(id_x).expect("no grad for x");
    let grad_x_real = WArr::from_vec1(vec![11.0f32, 7.0, 13.0])?;
    let y_data = y.read_data()?.clone();
    log::debug!("x id: {}, grads: {:?}", id_x, grads.keys());
    // y = x^2 + 5.x + 4
    assert_eq!(y_data, y_data_real, "y data");
    // dy/dx = 2.x + 5
    assert_eq!(grad_x, grad_x_real, "grad x");
    Ok(())
}



#[test]
fn test_grad2() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data_real: WArr<F32> = WArr::from_vec1_f64((0..12).collect::<Vec<_>>())?.reshape((2, 2, 3))?;
    let x2_data_real = WArr::from_vec1_f64((0..12).collect::<Vec<_>>())?.reshape((2, 3, 2))?;
    let x1 = WTensor::from_data(x1_data_real.clone(), true)?;
    let x2 = WTensor::from_data(x2_data_real.clone(), true)?;
    let id_x1 = x1.id();
    let id_x2 = x2.id();
    let y = x1.matmul(&x2)?;
    let mut grads = y.backward()?;
    
    let grad_x1 = grads.remove(id_x1).expect("no grad for x1");
    let grad_x2 = grads.remove(id_x2).expect("no grad for x1");
    let grad_x1_real = WArr::from_vec3_f64(vec![
        vec![vec![1.0, 5.0, 9.0], vec![1.0, 5.0, 9.0]],
        vec![vec![13.0, 17.0, 21.0], vec![13.0, 17.0, 21.0]],
    ])?;
    let grad_x2_real = WArr::from_vec3_f64(vec![
        vec![vec![3.0, 3.0], vec![5.0, 5.0], vec![7.0, 7.0]],
        vec![vec![15.0, 15.0], vec![17.0, 17.0], vec![19.0, 19.0]],
    ])?;
    let y_data_real = WArr::from_vec3_f64(vec![
        vec![vec![10.0, 13.0],vec![28.0, 40.0]],
        vec![vec![172.0, 193.0],vec![244.0, 274.0]]
    ])?;
    let y_data = y.read_data()?.clone();
    
    assert_eq!(y_data, y_data_real, "y data");
    assert_eq!(grad_x1, grad_x1_real, "grad x1");
    assert_eq!(grad_x2, grad_x2_real, "grad x1");
    Ok(())
}




#[test]
fn test_tensor_drop() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data_real: WArr<F32> = WArr::from_vec1_f64((0..12).collect::<Vec<_>>())?.reshape((2, 2, 3))?;
    let x2_data_real = WArr::from_vec1_f64((0..12).collect::<Vec<_>>())?.reshape((2, 3, 2))?;
    let x1 = WTensor::from_data(x1_data_real.clone(), true)?;
    let x2 = WTensor::from_data(x2_data_real.clone(), true)?;
    {
        let y = x1.matmul(&x2)?;
        let _ = y.backward()?;
    }
    {
        let y = x1.matmul(&x2)?;
        let _ = y.backward()?;
    }
    {
        let y = x1.matmul(&x2)?;
        let _ = y.backward()?;
    }
    Ok(())
}



#[test]
fn test_tensor_max() -> WResult<()> {
    let data = vec![vec![vec![3u32, 1, 4], vec![1, 5, 9]], vec![vec![2, 1, 7], vec![8, 2, 8]]];
    let data: WTensor<U32> = WTensor::from_vec3(data)?;

    let t0_real = WTensor::from_vec3(vec![vec![vec![4, ], vec![9, ]], vec![vec![7, ], vec![8, ]]])?;
    let t1_real = WTensor::from_vec3(vec![vec![vec![4, 4, 4], vec![9, 9, 9]], vec![vec![7, 7, 7], vec![8, 8, 8]]])?;

    let t0 = data.max(2).unwrap();
    let t1 = data.max_keepdim(2).unwrap();
    assert_eq!(t0, t0_real, "t0 max");
    assert_eq!(t1, t1_real, "t1 max");
    Ok(())
}



#[test]
fn test_tensor_sum() -> WResult<()> {
    let data = vec![vec![vec![3u32, 1, 4], vec![1, 5, 9]], vec![vec![2, 1, 7], vec![8, 2, 8]]];
    let data: WTensor<U32> = WTensor::from_vec3(data)?;

    let t0_real = WTensor::from_vec3(vec![vec![vec![8, ], vec![15, ]], vec![vec![10, ], vec![18, ]]])?;
    let t1_real = WTensor::from_vec3(vec![vec![vec![8, 8, 8], vec![15, 15, 15]], vec![vec![10, 10, 10], vec![18, 18, 18]]])?;

    let t0 = data.sum(2).unwrap();
    let t1 = data.sum_keepdim(2).unwrap();
    assert_eq!(t0, t0_real, "t0 sum");
    assert_eq!(t1, t1_real, "t1 sum");
    Ok(())
}



#[test]
fn test_tensor_exp() -> WResult<()> {
    let x1_data_real: WArr<F32> = WArr::from_vec1_f64((1..=12).collect::<Vec<_>>())?.reshape((4, 3))?;
    let x1 = WTensor::from_data(x1_data_real.clone(), true)?;

    #[allow(clippy::approx_constant)]
    #[allow(clippy::excessive_precision)]
    let x2 = WTensor::from_vec2(vec![vec![2.71828183,7.38905610,20.08553692],vec![54.59815003,148.41315910,403.42879349],vec![1096.63315843,2980.95798704,8103.08392758],vec![22026.46579481,59874.14171520,162754.79141900]])?;

    let t = x1.exp().unwrap();
    assert_eq!(t, x2, "exp");
    Ok(())
}



#[test]
fn test_tensor_grad_exp() -> WResult<()> {
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4., 0.15])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    #[allow(clippy::approx_constant)]
    let x2 = WArr::from_vec1(vec![20.0855, 2.7183, 54.5982, 1.1618])?.round(4.0)?;
    
    let id_x1 = x1.id();
    let y = x1.exp()?;
    let mut grads = y.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found").round(4.0)?;
    assert_eq!(x1_grad, x2);
    Ok(())
}



#[test]
fn test_tensor_grad_sqr() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x2 = WArr::from_vec1(vec![6., 2., 8.])?;
    
    let id_x1 = x1.id();
    let y = x1.sqr()?;
    let mut grads = y.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x2);
    Ok(())
}


#[test]
fn test_tensor_grad_mul() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x2 = WTensor::from_vec1(vec![4.0, 4., 4.])?;
    let x3 = WArr::from_vec1(vec![4.0, 4., 4.])?;
    let x4 = WTensor::from_vec1(vec![12., 4., 16.])?;
    
    let id_x1 = x1.id();

    let y = x1.mul(&x2)?;
    assert_eq!(y, x4, "res");

    let mut grads = y.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x3, "grad");
    
    Ok(())
}


#[test]
fn test_tensor_grad_braodcast_mul() -> WResult<()> {
    // log_init(false, false);
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x3 = WArr::from_vec1(vec![4.0, 4., 4.])?;
    let x4 = WTensor::from_vec1(vec![12., 4., 16.])?;
    
    let id_x1 = x1.id();

    let y = x1.broadcast_mul(&4.0)?;
    log::debug!("id: x1 {} {:?}, x4 {}, y {} {:?}", id_x1, x1, x4.id(), y.id(), y);

    assert_eq!(y, x4, "res");

    let mut grads = y.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x3, "grad");
    
    Ok(())
}


#[test]
fn test_tensor_grad_sum() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x3 = WTensor::from_vec1(vec![8.0])?;
    let x4 = WArr::from_vec1(vec![1.0, 1., 1.])?;
    let x5 = WTensor::from_vec1(vec![8.0, 8.0, 8.0])?;
    let x6 = WArr::from_vec1(vec![3.0, 3., 3.])?;
    
    let id_x1 = x1.id();

    let y1 = x1.sum(0)?;
    assert_eq!(y1, x3, "0 res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("0 x1 grad not found");
    assert_eq!(x1_grad, x4, "0 grad");

    
    let y1 = x1.sum_keepdim(0)?;
    assert_eq!(y1, x5, "1 res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("1 x1 grad not found");
    assert_eq!(x1_grad, x6, "1 grad");

    Ok(())
}




#[test]
fn test_tensor_grad_mix0() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x2 = WTensor::from_vec1(vec![26.0])?;
    let x3 = WArr::from_vec1(vec![6.0, 2.0, 8.0])?;
    
    let id_x1 = x1.id();

    let y1 = x1.sqr()?.sum(0)?;
    assert_eq!(y1, x2, "res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x3, "grad");

    Ok(())
}



#[test]
fn test_tensor_grad_mix1() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x2 = WTensor::from_vec1(vec![52.0])?;
    let x3 = WArr::from_vec1(vec![12.0, 4.0, 16.0])?;
    
    let id_x1 = x1.id();

    let y1 = x1.sqr()?.sum(0)?.broadcast_mul(&2.0)?;
    assert_eq!(y1, x2, "res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x3, "grad");

    Ok(())
}





#[test]
fn test_tensor_grad_max() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 1., 4.])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x3 = WTensor::from_vec1(vec![4.0])?;
    let x4 = WArr::from_vec1(vec![0., 0., 1.])?;
    let x5 = WTensor::from_vec1(vec![4.0, 4.0, 4.0])?;
    let x6 = WArr::from_vec1(vec![0., 0., 3.])?;
    
    let id_x1 = x1.id();

    let y1 = x1.max(0)?;
    assert_eq!(y1, x3, "0 res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("0 x1 grad not found");
    assert_eq!(x1_grad, x4, "0 grad");
    
    let y1 = x1.max_keepdim(0)?;
    assert_eq!(y1, x5, "1 res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("1 x1 grad not found");
    assert_eq!(x1_grad, x6, "1 grad");

    Ok(())
}



#[test]
fn test_tensor_grad_mean() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data: WArr<F32> = WArr::from_vec1(vec![3f32, 6., 9., 12.0])?;
    let x1 = WTensor::from_data(x1_data, true)?;
    let x3 = WTensor::from_vec1(vec![7.5])?;
    let x4 = WArr::from_vec1(vec![0.25, 0.25, 0.25, 0.25])?;
    
    let id_x1 = x1.id();

    let y1 = x1.mean(0)?;
    assert_eq!(y1, x3, "res no broad");

    let mut grads = y1.backward()?;
    let x1_grad = grads.remove(id_x1).expect("x1 grad not found");
    assert_eq!(x1_grad, x4, "grad");

    Ok(())
}


#[test]
fn test_grad_descent() -> WResult<()> {
    // log_init(true, false);
    
    let learning_rate = 0.1;
    let y_data: WArr<F32> = WArr::from_vec1(vec![4.199999f32])?;
    let y = WTensor::from_data(y_data, true)?;
    let x_data = WArr::from_vec1(vec![0.0f32])?;
    let x = WTensor::from_data(x_data, true)?;
    
    let id_x = x.id();
    for _step in 0..100 {
        let a = x.broadcast_sub(&4.2)?;
        let c = a.mul(&a)?;
        let mut grads = c.backward()?;
        let x_grad = grads.remove(id_x).expect("no grad for x");
        let grad_temp = x_grad.broadcast_mul(&learning_rate)?;
        let x_now = x.read_data()?.clone().sub(&grad_temp)?;
        x.write_data(x_now)?;
    }
    assert_eq!(x, y);
    Ok(())
}


#[test]
fn test_grad_mix2() -> WResult<()> {
    

    let x_data: WArr<F32> = WArr::from_vec1(vec![3.0f32, 1.0, 4.0, 0.15])?;
    let x = WTensor::from_data(x_data, true)?;
    let id_x = x.id();
    
    let y = x.ln()?.broadcast_add(&1.0)?;
    let mut grads = y.backward()?;
    let grad_x = grads.remove(id_x).expect("1 no grad for x").round(4.0)?;
    let y = y.read_data()?.round(4.0)?;
    let y_real = WArr::from_vec1(vec![2.0986, 1.0, 2.3863, -0.8971])?;
    let grad_x_real = WArr::from_vec1(vec![0.3333, 1.0, 0.25, 6.6667])?;
    assert_eq!(y, y_real, "1 res y");
    assert_eq!(grad_x, grad_x_real, "1 grad_x");
    

    let y = x.exp()?;
    let mut grads = y.backward()?;
    let grad_x = grads.remove(id_x).expect("2 no grad for x").round(4.0)?;
    let y = y.read_data()?.round(4.0)?;
    #[allow(clippy::approx_constant)]
    let y_real = WArr::from_vec1(vec![20.0855, 2.7183, 54.5982, 1.1618])?;
    #[allow(clippy::approx_constant)]
    let grad_x_real = WArr::from_vec1(vec![20.0855, 2.7183, 54.5982, 1.1618])?;
    assert_eq!(y, y_real, "2 res y");
    assert_eq!(grad_x, grad_x_real, "2 grad_x");


    let y = x.exp()?.sqr()?;
    let mut grads = y.backward()?;
    let grad_x = grads.remove(id_x).expect("3 no grad for x").round(2.0)?;
    let y = y.read_data()?.round(3.0)?;
    let y_real = WArr::from_vec1(vec![403.429, 7.389, 2980.958, 1.35])?;
    let grad_x_real = WArr::from_vec1(vec![806.86, 14.78, 5961.92, 2.7])?;
    assert_eq!(y, y_real, "3 res y");
    assert_eq!(grad_x, grad_x_real, "3 grad_x");
    
    Ok(())
}



#[test]
fn test_tensor_grad_broadcast() -> WResult<()> {
    let y_real: WTensor<F32> = WTensor::from_vec1(vec![6., 2., 8., 0.3])?;
    let b_real = WTensor::from_vec1(vec![0.5, 0.5, 0.5, 0.5])?;
    let x_data: WArr<F32> = WArr::from_vec1(vec![3.0f32, 1.0, 4.0, 0.15])?;
    let x = WTensor::from_data(x_data, true)?;
    let a = WTensor::from_vec1(vec![0.5f32])?;

    let b = a.broadcast(0, 4)?;
    assert_eq!(b_real, b, "b");

    let y = x.div(&b)?;
    assert_eq!(y_real, y, "y");

    let mut grads = y.backward()?;
    let idx = x.id();
    let grad_x = grads.remove(idx).expect("no grad for x");
    let grad_x_real = WArr::from_vec1(vec![2., 2., 2., 2.])?;
    assert_eq!(grad_x_real, grad_x, "grad x");

    Ok(())
}




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
        assert_eq!(ys, ys_log[step], "ys {}", step);
        
        let loss0 = ys.sub(&sample_ys)?.sqr()?;
        assert_eq!(loss0, loss0_log[step], "loss0 {}", step);

        let loss = loss0.sum_all()?;
        assert_eq!(loss, loss_log[step], "loss {}", step);
        
        sgd.backward(loss)?;
        assert_eq!(w, w_log[step], "w {}", step);
        assert_eq!(b, b_log[step], "b {}", step);
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
        assert_eq!(ys, ys_log[step], "ys {}", step);
        
        let loss = mse(&ys, &sample_ys)?;
        assert_eq!(loss, loss_log[step], "loss {}", step);
        
        sgd.backward(loss)?;
        assert_eq!(w, w_log[step], "w {}", step);
        assert_eq!(b, b_log[step], "b {}", step);
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


