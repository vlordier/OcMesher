use crate::CoreError;

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
        out.reserve(n_pts);
        for point in xyz.chunks_exact(3) {
            out.push(
                (point[0] * self.normal[0] + point[1] * self.normal[1] + point[2] * self.normal[2] - self.offset)
                    as f32,
            );
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
        out.reserve(n_pts);
        for point in xyz.chunks_exact(3) {
            let dx = point[0] - self.center[0];
            let dy = point[1] - self.center[1];
            let dz = point[2] - self.center[2];
            out.push(((dx * dx + dy * dy + dz * dz).sqrt() - self.radius) as f32);
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

    let mut full = Vec::with_capacity(n_pts * n_kernels);
    eval_sdf_full_native_into(kernels, xyz, n_pts, n_kernels, b_min, b_max, enclosed, &mut full)?;

    out.clear();
    out.resize(n_pts, f32::INFINITY);
    for i in 0..n_pts {
        for k in 0..n_kernels {
            let value = full[i * n_kernels + k];
            if value < out[i] {
                out[i] = value;
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
    use std::sync::Mutex;
    use tch::{Device, Kind, Tensor};

    const GPU_BATCH_THRESHOLD: usize = 131072;

    #[derive(Default)]
    struct TchScratch {
        xyz_f32: Vec<f32>,
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
        scratch: Mutex<TchScratch>,
    }

    impl TchPlaneKernel {
        pub fn new(normal: [f32; 3], offset: f32, device: Device) -> Result<Self, CoreError> {
            let normal = normalize_normal_f32(normal)?;
            Ok(Self {
                normal,
                offset,
                device,
                scratch: Mutex::new(TchScratch::default()),
            })
        }
    }

    pub struct TchSphereKernel {
        center: [f32; 3],
        radius: f32,
        device: Device,
        scratch: Mutex<TchScratch>,
    }

    impl TchSphereKernel {
        pub fn new(center: [f32; 3], radius: f32, device: Device) -> Self {
            Self {
                center,
                radius,
                device,
                scratch: Mutex::new(TchScratch::default()),
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

            if self.device == Device::Cpu || n_pts < GPU_BATCH_THRESHOLD {
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

            let xyz_tensor = {
                let mut scratch = self
                    .scratch
                    .lock()
                    .map_err(|_| CoreError::Sdf("TchSphereKernel scratch lock poisoned".to_string()))?;
                fill_xyz_f32_buffer(xyz, &mut scratch.xyz_f32);
                Tensor::from_slice(&scratch.xyz_f32)
                    .view([n_pts as i64, 3])
                    .to_device(self.device)
            };
            let center = Tensor::from_slice(&self.center)
                .view([1, 3])
                .to_device(self.device);
            let diff = xyz_tensor - center;
            let dims = [1i64];
            let dist = (&diff * &diff)
                .sum_dim_intlist(&dims[..], false, Kind::Float)
                .sqrt()
                - (self.radius as f64);
            let dist_cpu = dist.to_device(Device::Cpu);

            copy_tensor_to_vec(&dist_cpu, out);
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

            if self.device == Device::Cpu || n_pts < GPU_BATCH_THRESHOLD {
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

            let xyz_tensor = {
                let mut scratch = self
                    .scratch
                    .lock()
                    .map_err(|_| CoreError::Sdf("TchPlaneKernel scratch lock poisoned".to_string()))?;
                fill_xyz_f32_buffer(xyz, &mut scratch.xyz_f32);
                Tensor::from_slice(&scratch.xyz_f32)
                    .view([n_pts as i64, 3])
                    .to_device(self.device)
            };
            let normal = Tensor::from_slice(&self.normal)
                .view([3, 1])
                .to_device(self.device);
            let dist = xyz_tensor.matmul(&normal).squeeze_dim(1) - (self.offset as f64);
            let dist_cpu = dist.to_device(Device::Cpu);

            copy_tensor_to_vec(&dist_cpu, out);
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