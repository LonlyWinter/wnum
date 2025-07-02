use wnum::{array::{data::Data, error::WResult}, dtype::cuda::f32::F32};


#[test]
fn test_f32_new() -> WResult<()> {
    let data_raw1 = (0..10).map(| v | v as f32).collect::<Vec<_>>();
    let data_raw2 = (10..20).map(| v | v as f32).collect::<Vec<_>>();
    let data_raw3 = (0..20).map(| v | v as f32).collect::<Vec<_>>();
    
    let data1 = F32::from_slice(&data_raw1);
    let data2 = F32::from_slice(&data_raw2);

    let data3 = F32::from_slice_date(&[data1, data2]);
    let data3_res = data3.to_vec()?;

    assert_eq!(data_raw3, data3_res, "data3");
    
    Ok(())
}


#[test]
fn test_f32_drop() -> WResult<()> {
    let data_raw1 = (0..10).map(| v | v as f32).collect::<Vec<_>>();
    
    let data1 = F32::from_slice(&data_raw1);

    let data1_res = data1.to_vec()?;

    drop(data1);

    assert_eq!(data_raw1, data1_res, "data3");
    

    Ok(())
}