fn main() {
    if std::env::var_os("CARGO_FEATURE_TCH_KERNELS").is_none() {
        return;
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../torch/lib");
    }

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../torch/lib");
    }
}
