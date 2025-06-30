
#[derive(Debug, PartialEq)]
pub enum WError {
    ShapeNumMisMatch(u8, u8),
    DimMisMatch(usize, usize),
    DimNotFound(u8),
    DimNumError(String),
    DimCanotRepeat(u8, usize, usize),
    IDExist(String),
    Read(String),
    Cuda(String),
}




pub type WResult<T> = Result<T, WError>;