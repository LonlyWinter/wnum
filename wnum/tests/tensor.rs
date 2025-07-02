
use wnum::dtype::cpu::U32;
use wnum::dtype::cpu::F32;
#[allow(unused)]
use wnum::{array::{arr::WArr, error::WResult}, tensor::{base::WTensor, log::log_init}};


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
    assert_eq!(grad_x2, grad_x2_real, "grad x2");
    Ok(())
}



#[test]
fn test_grad_relu() -> WResult<()> {
    // log_init(false, false);
    
    let x1_data_real: WArr<F32> = WArr::from_vec3_f64(vec![vec![vec![-6.0, -5.0, -4.0], vec![-3.0, -2.0, -1.0]], vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]])?;
    let x1 = WTensor::from_data(x1_data_real.clone(), true)?;
    let id_x1 = x1.id();
    let y = x1.relu()?;
    let mut grads = y.sum_all()?.backward()?;
    
    let grad_x1 = grads.remove(id_x1).expect("no grad for x1");
    let grad_x1_real = WArr::from_vec3_f64(vec![vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]], vec![vec![0.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]])?;
    let y_data_real = WArr::from_vec3_f64(vec![vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]], vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]])?;
    let y_data = y.read_data()?.clone();
    
    assert_eq!(y_data, y_data_real, "y data");
    assert_eq!(grad_x1, grad_x1_real, "grad x1");
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
fn test_grad_gather_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![1.0, 0.0, 2.0]])?;
    let b = WTensor::from_data(b_data_real.clone(), false)?;
    
    let id_a = a.id();
    let y0 = a.gather(&b, 1)?;
    let y1 = y0.sum_all()?;
    let mut grads = y1.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let grad_a_real = WArr::from_vec2_f64(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]])?;
    let y0_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![4.0, 3.0, 5.0]])?;
    let y0_data = y0.read_data()?.clone();
    
    assert_eq!(y0_data, y0_real, "y data");
    assert_eq!(grad_a, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_gather_2() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![1.0, 0.0, 2.0]])?;
    let b = WTensor::from_data(b_data_real.clone(), false)?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![10.0, 11.0], vec![12.0, 13.0], vec![14.0, 15.0]])?;
    let c = WTensor::from_data(c_data_real.clone(), false)?;
    
    let id_a = a.id();
    let y0 = a.gather(&b, 1)?;
    let y1 = y0.matmul(&c)?.mean_all()?;
    let mut grads = y1.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let grad_a_real = WArr::from_vec2_f64(vec![vec![5.25, 7.25, 6.25], vec![6.25, 5.25, 7.25]])?;
    let y1_real = WArr::from_vec2_f64(vec![vec![95.75]])?;
    let y0_real = WArr::from_vec2_f64(vec![vec![0.0, 2.0, 1.0], vec![4.0, 3.0, 5.0]])?;
    let y1_data = y1.read_data()?.clone();
    let y0_data = y0.read_data()?.clone();
    
    assert_eq!(y1_data, y1_real, "y1 data");
    assert_eq!(y0_data, y0_real, "y0 data");
    assert_eq!(grad_a, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_ln_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec1_f64(vec![0.0000, 0.6931, 1.0986, 1.3863, 1.6094, 1.7918])?;
    let grad_a_real = WArr::from_vec1_f64(vec![1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667])?;
    let loss_data_real = WArr::from_vec1_f64(vec![6.5793])?;
    

    let id_a = a.id();
    let b = a.ln()?;
    let loss = b.sum_all()?;
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
fn test_grad_exp_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec1_f64(vec![2.718, 7.389, 20.086, 54.598, 148.413, 403.429])?;
    let grad_a_real = WArr::from_vec1_f64(vec![0.453, 1.232, 3.348, 9.100, 24.736, 67.238])?;
    let loss_data_real = WArr::from_vec1_f64(vec![106.106])?;
    

    let id_a = a.id();
    let b = a.exp()?;
    let loss = b.mean_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    assert_eq!(loss_data.round(3.0)?, loss_data_real, "loss data");
    assert_eq!(b_data.round(3.0)?, b_data_real, "b data");
    assert_eq!(grad_a.round(3.0)?, grad_a_real, "grad a");
    Ok(())
}


#[test]
fn test_grad_exp_2() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec1_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec1_f64(vec![2.718, 7.389, 20.086, 54.598, 148.413, 403.429])?;
    let grad_a_real = WArr::from_vec1_f64(vec![2.718, 7.389, 20.086, 54.598, 148.413, 403.429])?;
    let loss_data_real = WArr::from_vec1_f64(vec![636.633])?;
    

    let id_a = a.id();
    let b = a.exp()?;
    let loss = b.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let loss_data = loss.read_data()?.clone();
    let b_data = b.read_data()?.clone();
    assert_eq!(loss_data.round(3.0)?, loss_data_real, "loss data");
    assert_eq!(b_data.round(3.0)?, b_data_real, "b data");
    assert_eq!(grad_a.round(3.0)?, grad_a_real, "grad a");
    Ok(())
}


#[test]
fn test_grad_logsofmax_2() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, 1.0000], vec![0.1353, 0.3679, 1.0000]])?;
    let e_data_real = WArr::from_vec2_f64(vec![vec![1.5032], vec![1.5032]])?;
    let f_data_real = WArr::from_vec2_f64(vec![vec![0.4076], vec![0.4076]])?;
    let g_data_real = WArr::from_vec2_f64(vec![vec![-2.4076, -1.4076, -0.4076], vec![-2.4076, -1.4076, -0.4076]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.7299, 0.2658, -0.9957], vec![0.7299, 0.2658, -0.9957]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-8.4456]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = d.sum(1)?;
    let f = e.ln()?;
    let ff = f.broadcast(1, 3)?;
    let g = c.sub(&ff)?;
    let gg = a.log_softmax(1)?;
    let loss = g.sum_all()?;
    let loss_gg = gg.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(e.read_data()?.round(4.0)?, e_data_real, "e");
    assert_eq!(f.read_data()?.round(4.0)?, f_data_real, "f");
    assert_eq!(g.read_data()?.round(4.0)?, g_data_real, "g");
    assert_eq!(gg.read_data()?.round(4.0)?, g_data_real, "gg");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss g");
    assert_eq!(loss_gg.read_data()?.round(4.0)?, loss_data_real, "loss gg");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}


#[test]
fn test_grad_logsofmax_3() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, 1.0000], vec![0.1353, 0.3679, 1.0000]])?;
    let e_data_real = WArr::from_vec2_f64(vec![vec![1.5032], vec![1.5032]])?;
    let f_data_real = WArr::from_vec2_f64(vec![vec![0.4076], vec![0.4076]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.2701, 0.7342, -1.0043], vec![0.2701, 0.7342, -1.0043]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![2.4456]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = d.sum(1)?;
    let f = e.ln()?;
    let ff = f.broadcast(1, 3)?;
    let loss = ff.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(e.read_data()?.round(4.0)?, e_data_real, "e");
    assert_eq!(f.read_data()?.round(4.0)?, f_data_real, "f");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss g");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}


#[test]
fn test_grad_broadcast_2() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0], vec![1.0], vec![2.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![0.0000, 0.0000], vec![1.0000, 1.0000], vec![2.0000, 2.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![2.0000], vec![2.0000], vec![2.0000]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![6.0]])?;

    let id_a = a.id();
    let b = a.broadcast(1, 2)?;
    let loss = b.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_sub_1() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![1.0000, 1.0000, -2.0000], vec![1.0000, 1.0000, -2.0000]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-6.0]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let loss = c.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_grad_sub_2() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.8647, 0.6321, -1.4968], vec![0.8647, 0.6321, -1.4968]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-9.0064]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = c.sub(&d)?;
    let loss = e.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_3() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![1.0000, 1.0000, -2.0000], vec![1.0000, 1.0000, -2.0000]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-6.0]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let loss = c.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}





#[test]
fn test_grad_sub_4() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![1.0000, 2.7183, 7.3891], vec![20.0855, 54.5982, 148.4132]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-1.0000, -1.7183, -5.3891], vec![-17.0855, -50.5981, -143.4132]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.0000, -1.7183, -6.3891], vec![-19.0855, -53.5982, -147.4132]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-219.2042]])?;
    

    let id_a = a.id();
    let b = a.exp()?;
    let c = a.sub(&b)?;
    let loss = c.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_5() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, 1.0000], vec![0.1353, 0.3679, 1.0000]])?;
    let e_data_real = WArr::from_vec2_f64(vec![vec![1.5032, 1.5032, 1.5032], vec![1.5032, 1.5032, 1.5032]])?;
    let f_data_real = WArr::from_vec2_f64(vec![vec![0.0900, 0.2447, 0.6652], vec![0.0900, 0.2447, 0.6652]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.0000, 0.0000, 0.0000], vec![0.0000, 0.0000, 0.0000]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![2.0000]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = d.sum_keepdim(1)?;
    let f = d.div(&e)?;
    let ff = a.softmax(1)?;
    let loss = ff.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(e.read_data()?.round(4.0)?, e_data_real, "e");
    assert_eq!(f.read_data()?.round(4.0)?, f_data_real, "f");
    assert_eq!(ff.read_data()?.round(4.0)?, f_data_real, "ff");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_6() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, 1.0000], vec![0.1353, 0.3679, 1.0000]])?;
    let e_data_real = WArr::from_vec2_f64(vec![vec![1.5032], vec![1.5032]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, -0.5032], vec![0.1353, 0.3679, -0.5032]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![3.0064]])?;
    

    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = d.sum(1)?;
    let loss = e.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(e.read_data()?.round(4.0)?, e_data_real, "e");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_7() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![-2.0000, -1.0000, 0.0000], vec![-2.0000, -1.0000, 0.0000]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.1353, 0.3679, 1.0000], vec![0.1353, 0.3679, 1.0000]])?;
    let e_data_real = WArr::from_vec2_f64(vec![vec![-14.7781, -2.7183, 0.0000], vec![-14.7781, -2.7183, 0.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![22.1672, 5.4366, -27.6037], vec![22.1672, 5.4366, -27.6037]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![-34.9928]])?;
    
    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.sub(&b)?;
    let d = c.exp()?;
    let e = c.div(&d)?;
    let loss = e.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(e.read_data()?.round(4.0)?, e_data_real, "e");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_8() -> WResult<()> {
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![2.0000, 2.0000, 2.0000], vec![5.0000, 5.0000, 5.0000]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![0.0000, 0.5000, 1.0000], vec![0.6000, 0.8000, 1.0000]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.5000, 0.5000, -0.2500], vec![0.2000, 0.2000, -0.2800]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![3.9000]])?;
    
    let id_a = a.id();
    let b = a.max_keepdim(1)?;
    let c = a.div(&b)?;
    let loss = c.sum_all()?;
    let mut grads = loss.backward()?;
    
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}




#[test]
fn test_grad_sub_9() -> WResult<()> {
    log_init(true, true);
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![1.0000, 2.7183, 7.3891], vec![20.0855, 54.5982, 148.4132]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![0.0000, 0.3679, 0.2707], vec![0.1494, 0.0733, 0.0337]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![1.0000, -0.0000, -0.1353], vec![-0.0996, -0.0549, -0.0270]])?;
    let grad_b_real = WArr::from_vec2_f64(vec![vec![-0.0000, -0.1353, -0.0366], vec![-0.0074, -0.0013, -0.0002]])?;
    let grad_c_real = WArr::from_vec2_f64(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![0.8949]])?;
    
    let b = a.exp()?;
    let c = a.div(&b)?;
    let loss = c.sum_all()?;
    let mut grads = loss.backward()?;
    
    let id_a = a.id();
    let id_b = b.id();
    let id_c = c.id();
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let grad_b = grads.remove(id_b).expect("no grad for b");
    let grad_c = grads.remove(id_c).expect("no grad for c");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_c.round(4.0)?, grad_c_real, "grad c");
    assert_eq!(grad_b.round(4.0)?, grad_b_real, "grad b");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}



#[test]
fn test_tensor_grad_sqrt() -> WResult<()> {
    log_init(true, true);
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![1.0000, 1.4142, 1.7321], vec![2.0000, 2.2361, 2.4495]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![0.5000, 0.3536, 0.2887], vec![0.2500, 0.2236, 0.2041]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![10.8318]])?;
    
    let b = a.sqrt()?;
    let loss = b.sum_all()?;
    let mut grads = loss.backward()?;
    
    let id_a = a.id();
    let grad_a = grads.remove(id_a).expect("no grad for a");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}







#[test]
fn test_grad_concat_1() -> WResult<()> {
    log_init(true, true);
    let a_data_real: WArr<F32> = WArr::from_vec2_f64(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]])?;
    let a = WTensor::from_data(a_data_real.clone(), true)?;
    let b_data_real = WArr::from_vec2_f64(vec![vec![1.0000, 2.7183, 7.3891], vec![20.0855, 54.5982, 148.4132]])?;
    let c_data_real = WArr::from_vec2_f64(vec![vec![0.0000, 0.3679, 0.2707], vec![0.1494, 0.0733, 0.0337]])?;
    let d_data_real = WArr::from_vec2_f64(vec![vec![0.0000, 1.0000, 2.0000, 0.0000, 0.3679, 0.2707], vec![3.0000, 4.0000, 5.0000, 0.1494, 0.0733, 0.0337]])?;
    let grad_a_real = WArr::from_vec2_f64(vec![vec![2.0000, 1.0000, 0.8647], vec![0.9004, 0.9451, 0.9730]])?;
    let grad_b_real = WArr::from_vec2_f64(vec![vec![-0.0000, -0.1353, -0.0366], vec![-0.0074, -0.0013, -0.0002]])?;
    let grad_c_real = WArr::from_vec2_f64(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]])?;
    let grad_d_real = WArr::from_vec2_f64(vec![vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])?;
    let loss_data_real = WArr::from_vec2_f64(vec![vec![15.8949]])?;
    
    let b = a.exp()?;
    let c = a.div(&b)?;
    let d = WTensor::cat(vec![&a, &c], 1)?;
    let loss = d.sum_all()?;
    let mut grads = loss.backward()?;
    
    let id_a = a.id();
    let id_b = b.id();
    let id_c = c.id();
    let id_d = d.id();
    let grad_a = grads.remove(id_a).expect("no grad for a");
    let grad_b = grads.remove(id_b).expect("no grad for b");
    let grad_c = grads.remove(id_c).expect("no grad for c");
    let grad_d = grads.remove(id_d).expect("no grad for d");
    assert_eq!(b.read_data()?.round(4.0)?, b_data_real, "b");
    assert_eq!(c.read_data()?.round(4.0)?, c_data_real, "c");
    assert_eq!(d.read_data()?.round(4.0)?, d_data_real, "d");
    assert_eq!(loss.read_data()?.round(4.0)?, loss_data_real, "loss");
    assert_eq!(grad_d.round(4.0)?, grad_d_real, "grad d");
    assert_eq!(grad_c.round(4.0)?, grad_c_real, "grad c");
    assert_eq!(grad_b.round(4.0)?, grad_b_real, "grad b");
    assert_eq!(grad_a.round(4.0)?, grad_a_real, "grad a");
    Ok(())
}









