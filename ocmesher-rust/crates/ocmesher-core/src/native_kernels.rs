use crate::CoreError;

pub trait SdfEvaluator: Send + Sync {
    fn evaluate_batch(&self, xyz: &[f64], n_pts: usize) -> Result<Vec<f32>, CoreError>;
}

pub type BoxedSdfEvaluator = Box<dyn SdfEvaluator>;

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
    fn evaluate_batch(&self, xyz: &[f64], n_pts: usize) -> Result<Vec<f32>, CoreError> {
        if xyz.len() != n_pts * 3 {
            return Err(CoreError::Sdf(format!(
                "SphereKernel expected {} coordinates, got {}",
                n_pts * 3,
                xyz.len()
            )));
        }
        let mut out = Vec::with_capacity(n_pts);
        for point in xyz.chunks_exact(3) {
            let dx = point[0] - self.center[0];
            let dy = point[1] - self.center[1];
            let dz = point[2] - self.center[2];
            out.push(((dx * dx + dy * dy + dz * dz).sqrt() - self.radius) as f32);
        }
        Ok(out)
    }
}

pub(crate) fn eval_sdf_full_native(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    n_kernels: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> Result<Vec<f32>, CoreError> {
    let mut result = vec![0.0f32; n_pts * n_kernels];

    for (k_idx, kernel) in kernels.iter().enumerate() {
        let sdf = kernel.evaluate_batch(xyz, n_pts)?;
        if sdf.len() != n_pts {
            return Err(CoreError::Sdf(format!(
                "native kernel {k_idx} returned {} values for {n_pts} query points",
                sdf.len()
            )));
        }
        for (i, value) in sdf.iter().enumerate() {
            result[i * n_kernels + k_idx] = *value;
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
                    result[i * n_kernels + k] = 1.0;
                }
            }
        }
    }

    Ok(result)
}

pub(crate) fn eval_sdf_min_native(
    kernels: &[BoxedSdfEvaluator],
    xyz: &[f64],
    n_pts: usize,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> Result<Vec<f32>, CoreError> {
    let n_kernels = kernels.len();
    let full = eval_sdf_full_native(kernels, xyz, n_pts, n_kernels, b_min, b_max, enclosed)?;
    if n_kernels == 1 {
        return Ok(full);
    }

    let mut min_sdf = vec![f32::INFINITY; n_pts];
    for i in 0..n_pts {
        for k in 0..n_kernels {
            let value = full[i * n_kernels + k];
            if value < min_sdf[i] {
                min_sdf[i] = value;
            }
        }
    }
    Ok(min_sdf)
}

#[cfg(feature = "tch-kernels")]
pub mod tch_kernels {
    use super::SdfEvaluator;
    use crate::CoreError;
    use tch::{Device, Kind, Tensor};

    #[derive(Clone, Debug)]
    pub struct TchSphereKernel {
        center: [f32; 3],
        radius: f32,
        device: Device,
    }

    impl TchSphereKernel {
        pub fn new(center: [f32; 3], radius: f32, device: Device) -> Self {
            Self {
                center,
                radius,
                device,
            }
        }
    }

    impl SdfEvaluator for TchSphereKernel {
        fn evaluate_batch(&self, xyz: &[f64], n_pts: usize) -> Result<Vec<f32>, CoreError> {
            if xyz.len() != n_pts * 3 {
                return Err(CoreError::Sdf(format!(
                    "TchSphereKernel expected {} coordinates, got {}",
                    n_pts * 3,
                    xyz.len()
                )));
            }

            let xyz_f32: Vec<f32> = xyz.iter().map(|&value| value as f32).collect();
            let xyz_tensor = Tensor::from_slice(&xyz_f32)
                .view([n_pts as i64, 3])
                .to_device(self.device);
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
            let mut out = vec![0f32; dist_cpu.numel()];
            let len = out.len();
            dist_cpu.copy_data(&mut out, len);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoxedSdfEvaluator, SphereKernel, eval_sdf_full_native, eval_sdf_min_native};

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
}