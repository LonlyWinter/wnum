use std::{env, io};
use std::process::Command;


fn build_lib(dir: &str) -> io::Result<()> {
    let lib = dir.split("/").last().unwrap();
    
    let out_dir = env::var("OUT_DIR").unwrap();

    let status = Command::new("make")
        .arg("clean")
        .current_dir(dir)
        .status()?;

    if !status.success() {
        return Err(io::Error::other("run make clean first error"));
    }
    
    let status = Command::new("make")
        .arg(format!("DIR={out_dir}"))
        .current_dir(dir)
        .status()?;

    if !status.success() {
        return Err(io::Error::other("run make error"));
    }

    let status = Command::new("make")
        .arg("clean")
        .current_dir(dir)
        .status()?;

    if !status.success() {
        return Err(io::Error::other("run make clean final error"));
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=dylib={lib}");

    Ok(())
}



fn main() -> io::Result<()> {
    build_lib("src/dtype/cuda/f32")?;
    Ok(())
}