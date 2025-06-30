

pub mod dtype;

#[cfg(test)]
mod f32;


#[cfg(feature = "simd")]
pub mod simd;

// pub mod boolean;

pub use dtype::{F32, F64, U32, U8, I32};

