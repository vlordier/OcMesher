use crate::CoreError;
use rayon::prelude::*;

const PARALLEL_EVAL_THRESHOLD: usize = 32768;

pub trait SdfEvaluator: Send + Sync {
    fn evaluate_batch_into(&self, xyz: &[f64], n_pts: usize, out: &mut Vec<f32>) -> Result<(), CoreError>;

    fn evaluate_batch(&self, xyz: &[f64], n_pts: usize) -> Result<Vec<f32>, CoreError> {
        let mut out = Vec::with_capacity(n_pts);
        self.evaluate_batch_into(xyz, n_pts, &mut out)?;
        Ok(out)
    }
}

pub type BoxedSdfEvaluator = Box<dyn SdfEvaluator>;

#[derive(Clone, Debug)]
pub struct SphereSpec {
    pub center: [f64; 3],
    pub radius: f64,
}

impl SphereSpec {
    pub fn new(center: [f64; 3], radius: f64) -> Result<Self, CoreError> {
        if radius <= 0.0 {
            return Err(CoreError::Sdf("radius must be positive".to_string()));
        }
        Ok(Self { center, radius })
    }
}

#[derive(Clone, Debug)]
pub struct PlaneSpec {
    pub normal: [f64; 3],
    pub offset: f64,
}

impl PlaneSpec {
    pub fn new(normal: [f64; 3], offset: f64) -> Result<Self, CoreError> {
        let _ = normalize_normal(normal)?;
        Ok(Self { normal, offset })
    }
}

#[derive(Clone, Debug)]
pub enum PrimitiveSpec {
    Sphere(SphereSpec),
    Plane(PlaneSpec),
}

fn normalize_normal(normal: [f64; 3]) -> Result<[f64; 3], CoreError> {
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if norm <= f64::EPSILON {
        return Err(CoreError::Sdf("PlaneKernel normal must be non-zero".to_string()));
    }
    Ok([normal[0] / norm, normal[1] / norm, normal[2] / norm])
}

pub fn build_native_kernels(specs: &[PrimitiveSpec]) -> Result<Vec<BoxedSdfEvaluator>, CoreError> {
    let mut kernels = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            PrimitiveSpec::Sphere(sphere) => {
                kernels.push(Box::new(SphereKernel::new(sphere.center, sphere.radius)) as BoxedSdfEvaluator);
            }
            PrimitiveSpec::Plane(plane) => {
                kernels.push(Box::new(PlaneKernel::new(plane.normal, plane.offset)?) as BoxedSdfEvaluator);
            }
        }
    }
    Ok(kernels)
}

#[derive(Clone, Debug)]
pub struct PlaneKernel {
    normal: [f64; 3],
    offset: f64,
}

impl PlaneKernel {
    pub fn new(normal: [f64; 3], offset: f64) -> Result<Self, CoreError> {
        let normal = normalize_normal(normal)?;
        Ok(Self {
            normal,
            offset,
        })
    }
}

impl Default for PlaneKernel {
    fn default() -> Self {
        Self::new([0.0, 0.0, 1.0], 0.0).expect("default plane normal must be valid")
    }
}

impl SdfEvaluator for PlaneKernel {
    fn evaluate_batch_into(&self, xyz: &[f64], n_pts: usize, out: &mut Vec<f32>) -> Result<(), CoreError> {
        if xyz.len() != n_pts * 3 {
            return Err(CoreError::Sdf(format!(
                "PlaneKernel expected {} coordinates, got {}",
                n_pts * 3,
                xyz.len()
            )));
        }

        out.clear();
        out.resize(n_pts, 0.0);
        if n_pts >= PARALLEL_EVAL_THRESHOLD {
            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                let base = i * 3;
                *slot = (xyz[base] * self.normal[0]
                    + xyz[base + 1] * self.normal[1]
                    + xyz[base + 2] * self.normal[2]
                    - self.offset) as f32;
            });
        } else {
            for (i, point) in xyz.chunks_exact(3).enumerate() {
                out[i] =
                    (point[0] * self.normal[0] + point[1] * self.normal[1] + point[2] * self.normal[2] - self.offset)
                        as f32;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SphereKernel {
    center: [f64; 3],
    radius: f64,
}

impl SphereKernel {
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Default for SphereKernel {
    fn default() -> Self {
        Self::new([0.0, 0.0, 0.0], 1.0)
    }
}

impl SdfEvaluator for SphereKernel {
    fn evaluate_batch_into(&self, xyz: &[f64], n_pts: usize, out: &mut Vec<f32>) -> Result<(), CoreError> {
        if xyz.len() != n_pts * 3 {
            return Err(CoreError::Sdf(format!(
                "SphereKernel expected {} coordinates, got {}",
                n_pts * 3,
                xyz.len()
            )));
        }

        out.clear();
        out.resize(n_pts, 0.0);
        if n_pts >= PARALLEL_EVAL_THRESHOLD {
            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                let base = i * 3;
                let dx = xyz[base] - self.center[0];
                let dy = xyz[base + 1] - self.center[1];
                let dz = xyz[base + 2] - self.center[2];
                *slot = ((dx * dx + dy * dy + dz * dz).sqrt() - self.radius) as f32;
            });
        } else {
            for (i, point) in xyz.chunks_exact(3).enumerate() {
                let dx = point[0] - self.center[0];
                let dy = point[1] - self.center[1];
                let dz = point[2] - self.center[2];
                out[i] = ((dx * dx + dy * dy + dz * dz).sqrt() - self.radius) as f32;
            }
        }
        Ok(())
    }
}

pub(crate) fn eval_sdf_full_native_into(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    n_kernels: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
    out: &mut Vec<f32>,
) -> Result<(), CoreError> {
    if n_kernels == 1 {
        kernels[0].evaluate_batch_into(xyz, n_pts, out)?;
        if out.len() != n_pts {
            return Err(CoreError::Sdf(format!(
                "native kernel 0 returned {} values for {n_pts} query points",
                out.len()
            )));
        }

        if enclosed {
            for i in 0..n_pts {
                let x = xyz[i * 3];
                let y = xyz[i * 3 + 1];
                let z = xyz[i * 3 + 2];
                if x <= b_min[0]
                    || x >= b_max[0]
                    || y <= b_min[1]
                    || y >= b_max[1]
                    || z <= b_min[2]
                    || z >= b_max[2]
                {
                    out[i] = 1.0;
                }
            }
        }

        return Ok(());
    }

    out.clear();
    out.resize(n_pts * n_kernels, 0.0f32);
    let mut kernel_sdf = Vec::with_capacity(n_pts);

    for (k_idx, kernel) in kernels.iter().enumerate() {
        kernel.evaluate_batch_into(xyz, n_pts, &mut kernel_sdf)?;
        if kernel_sdf.len() != n_pts {
            return Err(CoreError::Sdf(format!(
                "native kernel {k_idx} returned {} values for {n_pts} query points",
                kernel_sdf.len()
            )));
        }
        for (i, value) in kernel_sdf.iter().enumerate() {
            out[i * n_kernels + k_idx] = *value;
        }
    }

    if enclosed {
        for i in 0..n_pts {
            let x = xyz[i * 3];
            let y = xyz[i * 3 + 1];
            let z = xyz[i * 3 + 2];
            if x <= b_min[0]
                || x >= b_max[0]
                || y <= b_min[1]
                || y >= b_max[1]
                || z <= b_min[2]
                || z >= b_max[2]
            {
                for k in 0..n_kernels {
                    out[i * n_kernels + k] = 1.0;
                }
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn eval_sdf_full_native(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    n_kernels: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> Result<Vec<f32>, CoreError> {
    let mut out = Vec::with_capacity(n_pts.max(n_pts * n_kernels));
    eval_sdf_full_native_into(kernels, xyz, n_pts, n_kernels, b_min, b_max, enclosed, &mut out)?;
    Ok(out)
}

pub(crate) fn eval_sdf_min_native_into(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
    out: &mut Vec<f32>,
) -> Result<(), CoreError> {
    let n_kernels = kernels.len();
    if n_kernels == 1 {
        kernels[0].evaluate_batch_into(xyz, n_pts, out)?;
        if out.len() != n_pts {
            return Err(CoreError::Sdf(format!(
                "native kernel 0 returned {} values for {n_pts} query points",
                out.len()
            )));
        }

        if enclosed {
            for i in 0..n_pts {
                let x = xyz[i * 3];
                let y = xyz[i * 3 + 1];
                let z = xyz[i * 3 + 2];
                if x <= b_min[0]
                    || x >= b_max[0]
                    || y <= b_min[1]
                    || y >= b_max[1]
                    || z <= b_min[2]
                    || z >= b_max[2]
                {
                    out[i] = 1.0;
                }
            }
        }

        return Ok(());
    }

    out.clear();
    out.resize(n_pts, f32::INFINITY);
    let mut kernel_out = Vec::with_capacity(n_pts);

    for (k_idx, kernel) in kernels.iter().enumerate() {
        kernel.evaluate_batch_into(xyz, n_pts, &mut kernel_out)?;
        if kernel_out.len() != n_pts {
            return Err(CoreError::Sdf(format!(
                "native kernel {k_idx} returned {} values for {n_pts} query points",
                kernel_out.len()
            )));
        }
        for (dst, value) in out.iter_mut().zip(kernel_out.iter()) {
            if *value < *dst {
                *dst = *value;
            }
        }
    }

    if enclosed {
        for i in 0..n_pts {
            let x = xyz[i * 3];
            let y = xyz[i * 3 + 1];
            let z = xyz[i * 3 + 2];
            if x <= b_min[0]
                || x >= b_max[0]
                || y <= b_min[1]
                || y >= b_max[1]
                || z <= b_min[2]
                || z >= b_max[2]
            {
                out[i] = 1.0;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn eval_sdf_min_native(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> Result<Vec<f32>, CoreError> {
    let mut out = Vec::with_capacity(n_pts);
    eval_sdf_min_native_into(kernels, xyz, n_pts, b_min, b_max, enclosed, &mut out)?;
    Ok(out)
}

#[cfg(feature = "tch-kernels")]
pub mod tch_kernels {
    use super::{BoxedSdfEvaluator, PrimitiveSpec, SdfEvaluator};
    use crate::CoreError;
    use std::sync::{Mutex, OnceLock};
    use tch::{Device, Kind, Tensor};

    const CUDA_BATCH_THRESHOLD: usize = 8192;
    const MPS_BATCH_THRESHOLD: usize = 32768;
    const CUDA_FP16_THRESHOLD: usize = 65536;
    const MPS_FP16_THRESHOLD: usize = 131072;
    const MPS_BF16_THRESHOLD: usize = 131072;
    const CUDA_FP8_THRESHOLD: usize = 262144;

    fn experimental_fp8_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("OCMESHER_TCH_EXPERIMENTAL_FP8")
                .ok()
                .as_deref()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
                .unwrap_or(false)
        })
    }

    fn mps_bf16_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("OCMESHER_TCH_MPS_BF16")
                .ok()
                .as_deref()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
                .unwrap_or(false)
        })
    }

    fn should_use_gpu(device: Device, n_pts: usize) -> bool {
        match device {
            Device::Cuda(_) => n_pts >= CUDA_BATCH_THRESHOLD,
            Device::Mps => n_pts >= MPS_BATCH_THRESHOLD,
            _ => false,
        }
    }

    fn gpu_compute_kind(device: Device, n_pts: usize) -> Kind {
        match device {
            Device::Cuda(_)
                if experimental_fp8_enabled() && n_pts >= CUDA_FP8_THRESHOLD =>
            {
                Kind::Float8e4m3fn
            }
            Device::Cuda(_) if n_pts >= CUDA_FP16_THRESHOLD => Kind::Half,
            Device::Mps if mps_bf16_enabled() && n_pts >= MPS_BF16_THRESHOLD => Kind::BFloat16,
            Device::Mps if n_pts >= MPS_FP16_THRESHOLD => Kind::Half,
            _ => Kind::Float,
        }
    }

    #[derive(Default)]
    struct TchIoScratch {
        xyz_f32: Vec<f32>,
        device_xyz: Option<Tensor>,
        device_rows: usize,
        device_xyz_half: Option<Tensor>,
        device_half_rows: usize,
        device_xyz_bf16: Option<Tensor>,
        device_bf16_rows: usize,
        device_xyz_fp8: Option<Tensor>,
        device_fp8_rows: usize,
        cpu_out: Option<Tensor>,
        cpu_rows: usize,
    }

    struct TchSphereScratch {
        io: TchIoScratch,
        center_row: Tensor,
        center_row_half: Option<Tensor>,
        center_row_bf16: Option<Tensor>,
        center_row_fp8: Option<Tensor>,
    }

    struct TchPlaneScratch {
        io: TchIoScratch,
        normal_row: Tensor,
        normal_row_half: Option<Tensor>,
        normal_row_bf16: Option<Tensor>,
        normal_row_fp8: Option<Tensor>,
    }

    fn fill_xyz_f32_buffer(xyz: &[f64], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(xyz.len().saturating_sub(out.capacity()));
        out.extend(xyz.iter().map(|value| *value as f32));
    }

    fn copy_tensor_to_vec(tensor: &Tensor, out: &mut Vec<f32>) {
        out.clear();
        out.resize(tensor.numel(), 0.0);
        let len = out.len();
        tensor.copy_data(out, len);
    }

    fn copy_from_device_into_vec(
        scratch: &mut TchIoScratch,
        device_tensor: &Tensor,
        n_pts: usize,
        out: &mut Vec<f32>,
    ) {
        if scratch.cpu_rows < n_pts {
            scratch.cpu_out = Some(Tensor::zeros([n_pts as i64], (Kind::Float, Device::Cpu)));
            scratch.cpu_rows = n_pts;
        }

        let cpu_view = scratch
            .cpu_out
            .as_ref()
            .expect("cpu output tensor should be initialized")
            .narrow(0, 0, n_pts as i64);
        let mut cpu_view_mut = cpu_view.shallow_clone();
        cpu_view_mut.copy_(device_tensor);
        copy_tensor_to_vec(&cpu_view, out);
    }

    fn load_xyz_to_device(
        scratch: &mut TchIoScratch,
        xyz: &[f64],
        n_pts: usize,
        device: Device,
        kind: Kind,
    ) -> Tensor {
        fill_xyz_f32_buffer(xyz, &mut scratch.xyz_f32);
        let xyz_cpu = Tensor::from_slice(&scratch.xyz_f32).view([n_pts as i64, 3]);

        let xyz_device = if kind == Kind::Half {
            if scratch.device_half_rows < n_pts {
                scratch.device_xyz_half = Some(Tensor::zeros([n_pts as i64, 3], (Kind::Half, device)));
                scratch.device_half_rows = n_pts;
            }
            scratch
                .device_xyz_half
                .as_ref()
                .expect("half device tensor should be initialized")
                .narrow(0, 0, n_pts as i64)
        } else if kind == Kind::BFloat16 {
            if scratch.device_bf16_rows < n_pts {
                scratch.device_xyz_bf16 = Some(Tensor::zeros([n_pts as i64, 3], (Kind::BFloat16, device)));
                scratch.device_bf16_rows = n_pts;
            }
            scratch
                .device_xyz_bf16
                .as_ref()
                .expect("bf16 device tensor should be initialized")
                .narrow(0, 0, n_pts as i64)
        } else if kind == Kind::Float8e4m3fn {
            if scratch.device_fp8_rows < n_pts {
                scratch.device_xyz_fp8 = Some(Tensor::zeros([n_pts as i64, 3], (Kind::Float8e4m3fn, device)));
                scratch.device_fp8_rows = n_pts;
            }
            scratch
                .device_xyz_fp8
                .as_ref()
                .expect("fp8 device tensor should be initialized")
                .narrow(0, 0, n_pts as i64)
        } else {
            if scratch.device_rows < n_pts {
                scratch.device_xyz = Some(Tensor::zeros([n_pts as i64, 3], (Kind::Float, device)));
                scratch.device_rows = n_pts;
            }
            scratch
                .device_xyz
                .as_ref()
                .expect("device tensor should be initialized")
                .narrow(0, 0, n_pts as i64)
        };

        let mut xyz_device_view = xyz_device.shallow_clone();
        xyz_device_view.copy_(&xyz_cpu);
        xyz_device
    }

    fn normalize_normal_f32(normal: [f32; 3]) -> Result<[f32; 3], CoreError> {
        let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm <= f32::EPSILON {
            return Err(CoreError::Sdf("TchPlaneKernel normal must be non-zero".to_string()));
        }
        Ok([normal[0] / norm, normal[1] / norm, normal[2] / norm])
    }

    pub fn build_tch_kernels(specs: &[PrimitiveSpec], device: Device) -> Result<Vec<BoxedSdfEvaluator>, CoreError> {
        let mut kernels = Vec::with_capacity(specs.len());
        for spec in specs {
            match spec {
                PrimitiveSpec::Sphere(sphere) => {
                    kernels.push(
                        Box::new(TchSphereKernel::new(
                            [sphere.center[0] as f32, sphere.center[1] as f32, sphere.center[2] as f32],
                            sphere.radius as f32,
                            device,
                        )) as BoxedSdfEvaluator,
                    );
                }
                PrimitiveSpec::Plane(plane) => {
                    kernels.push(
                        Box::new(TchPlaneKernel::new(
                            [plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32],
                            plane.offset as f32,
                            device,
                        )?) as BoxedSdfEvaluator,
                    );
                }
            }
        }
        Ok(kernels)
    }

    pub struct TchPlaneKernel {
        normal: [f32; 3],
        offset: f32,
        device: Device,
        scratch: Mutex<TchPlaneScratch>,
    }

    impl TchPlaneKernel {
        pub fn new(normal: [f32; 3], offset: f32, device: Device) -> Result<Self, CoreError> {
            let normal = normalize_normal_f32(normal)?;
            let normal_row_tensor = Tensor::from_slice(&normal)
                .view([1, 3])
                .to_device(device);
            Ok(Self {
                normal,
                offset,
                device,
                scratch: Mutex::new(TchPlaneScratch {
                    io: TchIoScratch::default(),
                    normal_row: normal_row_tensor,
                    normal_row_half: None,
                    normal_row_bf16: None,
                    normal_row_fp8: None,
                }),
            })
        }
    }

    pub struct TchSphereKernel {
        center: [f32; 3],
        radius: f32,
        device: Device,
        scratch: Mutex<TchSphereScratch>,
    }

    impl TchSphereKernel {
        pub fn new(center: [f32; 3], radius: f32, device: Device) -> Self {
            let center_tensor = Tensor::from_slice(&center)
                .view([1, 3])
                .to_device(device);
            Self {
                center,
                radius,
                device,
                scratch: Mutex::new(TchSphereScratch {
                    io: TchIoScratch::default(),
                    center_row: center_tensor,
                    center_row_half: None,
                    center_row_bf16: None,
                    center_row_fp8: None,
                }),
            }
        }
    }

    impl SdfEvaluator for TchSphereKernel {
        fn evaluate_batch_into(&self, xyz: &[f64], n_pts: usize, out: &mut Vec<f32>) -> Result<(), CoreError> {
            if xyz.len() != n_pts * 3 {
                return Err(CoreError::Sdf(format!(
                    "TchSphereKernel expected {} coordinates, got {}",
                    n_pts * 3,
                    xyz.len()
                )));
            }

            if !should_use_gpu(self.device, n_pts) {
                out.clear();
                out.reserve(n_pts);
                for point in xyz.chunks_exact(3) {
                    let dx = point[0] as f32 - self.center[0];
                    let dy = point[1] as f32 - self.center[1];
                    let dz = point[2] as f32 - self.center[2];
                    out.push((dx * dx + dy * dy + dz * dz).sqrt() - self.radius);
                }
                return Ok(());
            }

            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| CoreError::Sdf("TchSphereKernel scratch lock poisoned".to_string()))?;
            let compute_kind = gpu_compute_kind(self.device, n_pts);
            let xyz_tensor = load_xyz_to_device(&mut scratch.io, xyz, n_pts, self.device, compute_kind);
            let center_tensor = if compute_kind == Kind::Half {
                if scratch.center_row_half.is_none() {
                    scratch.center_row_half = Some(scratch.center_row.to_kind(Kind::Half));
                }
                scratch
                    .center_row_half
                    .as_ref()
                    .expect("half center tensor should be initialized")
            } else if compute_kind == Kind::BFloat16 {
                if scratch.center_row_bf16.is_none() {
                    scratch.center_row_bf16 = Some(scratch.center_row.to_kind(Kind::BFloat16));
                }
                scratch
                    .center_row_bf16
                    .as_ref()
                    .expect("bf16 center tensor should be initialized")
            } else if compute_kind == Kind::Float8e4m3fn {
                if scratch.center_row_fp8.is_none() {
                    scratch.center_row_fp8 = Some(scratch.center_row.to_kind(Kind::Float8e4m3fn));
                }
                scratch
                    .center_row_fp8
                    .as_ref()
                    .expect("fp8 center tensor should be initialized")
            } else {
                &scratch.center_row
            };
            let diff = xyz_tensor - center_tensor;
            let dims = [1i64];
            let dist = (&diff * &diff)
                .sum_dim_intlist(&dims[..], false, compute_kind)
                .sqrt()
                - (self.radius as f64);
            copy_from_device_into_vec(&mut scratch.io, &dist, n_pts, out);
            Ok(())
        }
    }

    impl SdfEvaluator for TchPlaneKernel {
        fn evaluate_batch_into(&self, xyz: &[f64], n_pts: usize, out: &mut Vec<f32>) -> Result<(), CoreError> {
            if xyz.len() != n_pts * 3 {
                return Err(CoreError::Sdf(format!(
                    "TchPlaneKernel expected {} coordinates, got {}",
                    n_pts * 3,
                    xyz.len()
                )));
            }

            if !should_use_gpu(self.device, n_pts) {
                out.clear();
                out.reserve(n_pts);
                for point in xyz.chunks_exact(3) {
                    out.push(
                        point[0] as f32 * self.normal[0]
                            + point[1] as f32 * self.normal[1]
                            + point[2] as f32 * self.normal[2]
                            - self.offset,
                    );
                }
                return Ok(());
            }

            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| CoreError::Sdf("TchPlaneKernel scratch lock poisoned".to_string()))?;
            let compute_kind = gpu_compute_kind(self.device, n_pts);
            let xyz_tensor = load_xyz_to_device(&mut scratch.io, xyz, n_pts, self.device, compute_kind);
            let normal_tensor = if compute_kind == Kind::Half {
                if scratch.normal_row_half.is_none() {
                    scratch.normal_row_half = Some(scratch.normal_row.to_kind(Kind::Half));
                }
                scratch
                    .normal_row_half
                    .as_ref()
                    .expect("half normal tensor should be initialized")
            } else if compute_kind == Kind::BFloat16 {
                if scratch.normal_row_bf16.is_none() {
                    scratch.normal_row_bf16 = Some(scratch.normal_row.to_kind(Kind::BFloat16));
                }
                scratch
                    .normal_row_bf16
                    .as_ref()
                    .expect("bf16 normal tensor should be initialized")
            } else if compute_kind == Kind::Float8e4m3fn {
                if scratch.normal_row_fp8.is_none() {
                    scratch.normal_row_fp8 = Some(scratch.normal_row.to_kind(Kind::Float8e4m3fn));
                }
                scratch
                    .normal_row_fp8
                    .as_ref()
                    .expect("fp8 normal tensor should be initialized")
            } else {
                &scratch.normal_row
            };
            let dims = [1i64];
            let dist = (&xyz_tensor * normal_tensor)
                .sum_dim_intlist(&dims[..], false, compute_kind)
                - (self.offset as f64);
            copy_from_device_into_vec(&mut scratch.io, &dist, n_pts, out);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BoxedSdfEvaluator, PlaneKernel, PlaneSpec, PrimitiveSpec, SphereKernel, SphereSpec,
        build_native_kernels, eval_sdf_full_native, eval_sdf_min_native,
    };

    #[test]
    fn build_native_kernels_constructs_scene() {
        let specs = vec![
            PrimitiveSpec::Sphere(SphereSpec::new([0.0, 0.0, 0.0], 1.0).unwrap()),
            PrimitiveSpec::Plane(PlaneSpec::new([0.0, 0.0, 1.0], 0.0).unwrap()),
        ];
        let kernels = build_native_kernels(&specs).unwrap();
        assert_eq!(kernels.len(), 2);
    }

    #[test]
    fn sphere_spec_rejects_non_positive_radius() {
        let err = SphereSpec::new([0.0, 0.0, 0.0], 0.0).unwrap_err();
        assert_eq!(err.to_string(), "SDF kernel call failed: radius must be positive");
    }

    #[test]
    fn plane_kernel_matches_expected_distances() {
        let kernel = PlaneKernel::new([0.0, 0.0, 1.0], 0.5).unwrap();
        let xyz = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let out = crate::native_kernels::SdfEvaluator::evaluate_batch(&kernel, &xyz, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] + 0.5).abs() <= 1e-6);
        assert!((out[1] - 1.5).abs() <= 1e-6);
    }

    #[test]
    fn sphere_kernel_matches_expected_distances() {
        let kernel = SphereKernel::new([0.0, 0.0, 0.0], 1.0);
        let xyz = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let out = crate::native_kernels::SdfEvaluator::evaluate_batch(&kernel, &xyz, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] + 1.0).abs() <= 1e-6);
        assert!((out[1] - 1.0).abs() <= 1e-6);
    }

    #[test]
    fn native_eval_applies_bounds_mask() {
        let kernels: Vec<BoxedSdfEvaluator> = vec![Box::new(SphereKernel::default())];
        let xyz = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let full = eval_sdf_full_native(&kernels, &xyz, 2, 1, &[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0], true)
            .unwrap();
        assert_eq!(full[1], 1.0);

        let min = eval_sdf_min_native(&kernels, &xyz, 2, &[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0], true).unwrap();
        assert_eq!(min[1], 1.0);
    }

    #[test]
    fn native_eval_min_combines_plane_and_sphere() {
        let kernels: Vec<BoxedSdfEvaluator> = vec![
            Box::new(SphereKernel::new([0.0, 0.0, 0.0], 1.0)),
            Box::new(PlaneKernel::new([0.0, 0.0, 1.0], 0.0).unwrap()),
        ];
        let xyz = [0.0, 0.0, 2.0, 0.0, 0.0, -2.0];
        let min = eval_sdf_min_native(&kernels, &xyz, 2, &[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0], false).unwrap();
        assert_eq!(min.len(), 2);
        assert!((min[0] - 1.0).abs() <= 1e-6);
        assert!((min[1] + 2.0).abs() <= 1e-6);
    }

    #[cfg(feature = "tch-kernels")]
    #[test]
    fn tch_sphere_kernel_matches_expected_distances() {
        use super::tch_kernels::TchSphereKernel;
        use tch::Device;

        let kernel = TchSphereKernel::new([0.0, 0.0, 0.0], 1.0, Device::Cpu);
        let xyz = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let out = crate::native_kernels::SdfEvaluator::evaluate_batch(&kernel, &xyz, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] + 1.0).abs() <= 1e-5);
        assert!((out[1] - 1.0).abs() <= 1e-5);
    }

    #[cfg(feature = "tch-kernels")]
    #[test]
    fn tch_plane_kernel_matches_expected_distances() {
        use super::tch_kernels::TchPlaneKernel;
        use tch::Device;

        let kernel = TchPlaneKernel::new([0.0, 0.0, 1.0], 0.5, Device::Cpu).unwrap();
        let xyz = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let out = crate::native_kernels::SdfEvaluator::evaluate_batch(&kernel, &xyz, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] + 0.5).abs() <= 1e-5);
        assert!((out[1] - 1.5).abs() <= 1e-5);
    }

    #[cfg(feature = "tch-kernels")]
    #[test]
    fn tch_eval_min_combines_plane_and_sphere() {
        use super::tch_kernels::{TchPlaneKernel, TchSphereKernel};
        use tch::Device;

        let kernels: Vec<BoxedSdfEvaluator> = vec![
            Box::new(TchSphereKernel::new([0.0, 0.0, 0.0], 1.0, Device::Cpu)),
            Box::new(TchPlaneKernel::new([0.0, 0.0, 1.0], 0.0, Device::Cpu).unwrap()),
        ];
        let xyz = [0.0, 0.0, 2.0, 0.0, 0.0, -2.0];
        let min = eval_sdf_min_native(&kernels, &xyz, 2, &[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0], false).unwrap();
        assert_eq!(min.len(), 2);
        assert!((min[0] - 1.0).abs() <= 1e-5);
        assert!((min[1] + 2.0).abs() <= 1e-5);
    }

    #[cfg(feature = "tch-kernels")]
    #[test]
    fn build_tch_kernels_constructs_scene() {
        use super::tch_kernels::build_tch_kernels;
        use tch::Device;

        let specs = vec![
            PrimitiveSpec::Sphere(SphereSpec::new([0.0, 0.0, 0.0], 1.0).unwrap()),
            PrimitiveSpec::Plane(PlaneSpec::new([0.0, 0.0, 1.0], 0.0).unwrap()),
        ];
        let kernels = build_tch_kernels(&specs, Device::Cpu).unwrap();
        assert_eq!(kernels.len(), 2);
    }
}