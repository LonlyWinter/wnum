use std::time::Instant;

use rand::{rng, Rng};
use rand_distr::StandardUniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "simd")]
use crate::{array::{arr::WArr, error::WResult}, tensor::base::WTensor};



fn normal(a: (usize, usize), b: (usize, usize), data_a: &[f32], data_b: &[f32]) {
    let s = a.1;
    let _ = (0..a.0).flat_map(| i | {
        let i0 = i*s;
        (0..b.1).map(move | j |{
            (0..s).map(move | i |{
                let a = data_a[i0 + i];
                let b = data_b[j + i * b.1];
                a * b
            }).sum::<f32>()
        })
    }).collect::<Vec<_>>();
}


fn parallel(a: (usize, usize), b: (usize, usize), data_a: &[f32], data_b: &[f32]) {
    let s = a.1;
    let _ = (0..a.0).into_par_iter().flat_map(| ii | {
        (0..b.1).into_par_iter().map(move | j |{
            (0..s).into_par_iter().map(move | i |{
                let a = data_a[ii*s + i];
                let b = data_b[j + i * b.1];
                a * b
            }).sum::<f32>()
        })
    }).collect::<Vec<_>>();
}

fn test_f<F: Fn((usize, usize), (usize, usize), &[f32], &[f32])>(tag: &str, a: usize, b: usize, c: usize, f: F) {
    let aa = (a, b);
    let bb = (b, c);
    let data_a = rng().sample_iter(StandardUniform).take(aa.0 * aa.1).collect::<Vec<f32>>();
    let data_b = rng().sample_iter(StandardUniform).take(bb.0 * bb.1).collect::<Vec<f32>>();
    let time_now = Instant::now();
    f(aa, bb, &data_a, &data_b);
    println!("{}\t{}\t{:?}\t{}/{}/{}", tag, a*b*c, time_now.elapsed().as_micros(), a, b, c);

}


#[allow(unused)]
#[test]
fn test_par_matmul() {
    println!("Method\tDimAll\t\tTime\tDims");
    let a = 256;
    let b = 1024;
    let c = 128;
    test_f("Normal", a, b, c, normal);
    test_f("Parall", a, b, c, parallel);
}


#[cfg(feature = "simd")]
fn vec_float_eq(a: &[f32], b: &[f32], acc: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(| (aa, bb) | (aa - bb).abs() < acc)
}


#[allow(unused)]
#[cfg(feature = "simd")]
#[test]
fn test_simd_matmul_1d() -> WResult<()> {
    let a = 4096;
    let b = 4096;
    let acc = 0.0001;
    println!("Method\tDimAll\tTime\tDims");

    let a_data = WArr::<crate::dtype::cpu::F32>::random_normal(a)?.to_vec()?;
    let b_data = WArr::<crate::dtype::cpu::F32>::random_normal(b)?.to_vec()?;


    let aa: WTensor<crate::dtype::cpu::simd::F32> = WTensor::from_vec1(a_data.clone())?;
    let bb = WTensor::from_vec1(b_data.clone())?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("SIMD\t{}\t{:?}\t{}/{}", a*b, time_now.elapsed().as_micros(), a, b);
    drop(aa);
    drop(bb);
    let c1 = cc.to_vec()?;
    drop(cc);

    let aa: WTensor<crate::dtype::cuda::F32> = WTensor::from_vec1(a_data.clone())?;
    let bb = WTensor::from_vec1(b_data.clone())?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("CUDA\t{}\t{:?}\t{}/{}", a*b, time_now.elapsed().as_micros(), a, b);
    drop(aa);
    drop(bb);
    let c2 = cc.to_vec()?;
    drop(cc);

    assert!(vec_float_eq(&c1, &c2, acc), "c1 vs c2");

    let aa: WTensor<crate::dtype::cpu::F32> = WTensor::from_vec1(a_data)?;
    let bb = WTensor::from_vec1(b_data)?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("FOR\t{}\t{:?}\t{}/{}", a*b, time_now.elapsed().as_micros(), a, b);
    drop(aa);
    drop(bb);
    let c3 = cc.to_vec()?;
    drop(cc);

    assert!(vec_float_eq(&c1, &c3, acc), "c1 vs c3");

    Ok(())
}



#[allow(unused)]
#[cfg(feature = "simd")]
#[test]
fn test_simd_matmul_2d() -> WResult<()> {
    let a = 256;
    let b = 1024;
    let c = 512;
    let acc = 0.01;
    println!("Method\tDimAll\tTime\tDims");

    let a_data = WArr::<crate::dtype::cpu::F32>::random_normal((a, b))?.broadcast_mul(&1000.0)?.to_vec()?;
    let b_data = WArr::<crate::dtype::cpu::F32>::random_normal((b, c))?.broadcast_mul(&1000.0)?.to_vec()?;


    let aa: WArr<crate::dtype::cpu::simd::F32> = WArr::from_vec1(a_data.clone())?.reshape((a, b))?;
    let bb = WArr::from_vec1(b_data.clone())?.reshape((b, c))?;
    let aa = WTensor::from_data(aa, false)?;
    let bb = WTensor::from_data(bb, false)?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("SIMD\t{}\t{:?}\t{}/{}/{}", a*b*c, time_now.elapsed().as_micros(), a, b, c);
    drop(aa);
    drop(bb);
    let c1 = cc.to_vec()?;
    drop(cc);

    let aa: WArr<crate::dtype::cuda::F32> = WArr::from_vec1(a_data.clone())?.reshape((a, b))?;
    let bb = WArr::from_vec1(b_data.clone())?.reshape((b, c))?;
    let aa = WTensor::from_data(aa, false)?;
    let bb = WTensor::from_data(bb, false)?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("CUDA\t{}\t{:?}\t{}/{}/{}", a*b*c, time_now.elapsed().as_micros(), a, b, c);
    drop(aa);
    drop(bb);
    let c2 = cc.to_vec()?;
    drop(cc);

    assert!(vec_float_eq(&c1, &c2, acc), "c1 vs c2\n{:?}\n{:?}", &c1, &c2);

    let aa: WArr<crate::dtype::cpu::F32> = WArr::from_vec1(a_data.clone())?.reshape((a, b))?;
    let bb = WArr::from_vec1(b_data.clone())?.reshape((b, c))?;
    let aa = WTensor::from_data(aa, false)?;
    let bb = WTensor::from_data(bb, false)?;
    let time_now = Instant::now();
    let cc = aa.matmul(&bb)?;
    println!("FOR\t{}\t{:?}\t{}/{}/{}", a*b*c, time_now.elapsed().as_micros(), a, b, c);
    drop(aa);
    drop(bb);
    let c3 = cc.to_vec()?;
    drop(cc);

    assert!(vec_float_eq(&c1, &c3, acc), "c1 vs c3\n{:?}\n{:?}", &c1, &c3);

    Ok(())
}