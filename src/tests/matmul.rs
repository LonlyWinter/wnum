use std::time::Instant;

use rand::{rng, Rng};
use rand_distr::StandardUniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};



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

fn test_f<F: Fn((usize, usize), (usize, usize), &[f32], &[f32])>(tag: &str, n: usize, a: usize, b: usize, c: usize, f: F) {
    let aa = (a, b);
    let bb = (b, c);
    let data_a = rng().sample_iter(StandardUniform).take(aa.0 * aa.1).collect::<Vec<f32>>();
    let data_b = rng().sample_iter(StandardUniform).take(bb.0 * bb.1).collect::<Vec<f32>>();
    let time_now = Instant::now();
    for _ in 0..n {
        f(aa, bb, &data_a, &data_b);
    }
    println!("{}\t{}\t{:?}\t{}/{}/{}", a*b*c, tag, time_now.elapsed().as_micros(), a, b, c);

}


#[allow(unused)]
// #[test]
fn test_par_matmul() {
    for i in 1..16 {
        for j in 8..16 {
            for k in 1..16 {
                test_f("normal", 100, i*64, j*64, k*64, normal);
                test_f("parall", 100, i*64, j*64, k*64, parallel);
            }
        }
    }
}