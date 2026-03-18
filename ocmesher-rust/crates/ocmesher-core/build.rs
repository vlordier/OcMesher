fn main() {
    if std::env::var_os("CARGO_FEATURE_TCH_KERNELS").is_none() {
        return;
    }

    let Some(libtorch_root) = std::env::var_os("LIBTORCH") else {
        return;
    };
    let libtorch_root = std::path::PathBuf::from(libtorch_root);
    let lib_dir = libtorch_root.join("lib");

    if !lib_dir.is_dir() {
        return;
    }

    if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}
