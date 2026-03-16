// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

//! PyO3 extension module `ocmesher_rust`.
//!
//! Exposes:
//! - [`Backend`]: Python class implementing [`RustBackendProtocol`] from `ocmesher.rust_backend`.
//!
//! Usage from Python:
//! ```python
//! import ocmesher_rust
//! from ocmesher import RustOcMesher
//!
//! backend = ocmesher_rust.Backend(
//!     lib_path="/path/to/ocmesher/lib/core.so",
//!     cameras=cameras,
//!     bounds=bounds,
//! )
//! mesher = RustOcMesher(cameras, bounds, backend=backend)
//! meshes, in_view_tags = mesher(sdf_kernels)
//! ```

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use ocmesher_core::{
    pack_cameras, run_meshing_pipeline, run_meshing_pipeline_native, validate_bounds, CoreLib,
    MesherParams, SphereKernel,
};
#[cfg(feature = "tch-kernels")]
use ocmesher_core::tch_kernels::TchSphereKernel;
#[cfg(feature = "tch-kernels")]
use tch::Device;

// ---------------------------------------------------------------------------
// Backend PyO3 class
// ---------------------------------------------------------------------------

/// Rust-backed OcMesher engine implementing RustBackendProtocol.
///
/// Constructor parameters
/// ----------------------
/// lib_path : str
///     Path to the compiled ``core.so`` shared library.
/// cameras : tuple[list, list, list[int], list[int]]
///     ``(cam_poses, Ks, Hs, Ws)`` — same format as :class:`OcMesher`.
/// bounds : list[float]
///     ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
/// pixels_per_cube, inv_scale, min_dist, memory_limit_mb,
/// bisection_iters, bisection_tol, enclosed, simplify_occluded,
/// visible_relax_iter, coarse_count
///     Forwarded verbatim to the C++ core (same defaults as ``OcMesher``).
#[pyclass(name = "Backend")]
pub struct Backend {
    lib: Arc<CoreLib>,
    params: MesherParams,
    version: String,
    supports_cuda: bool,
    supports_mps: bool,
}

#[pymethods]
impl Backend {
    #[new]
    #[pyo3(signature = (
        lib_path,
        cameras,
        bounds,
        pixels_per_cube = 8.0,
        inv_scale = 10.0,
        min_dist = 1.0,
        memory_limit_mb = 1000,
        bisection_iters = 15,
        bisection_tol = 0.0,
        enclosed = true,
        simplify_occluded = true,
        visible_relax_iter = 2,
        coarse_count = 500000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        lib_path: &str,
        cameras: &Bound<'_, PyTuple>,
        bounds: &Bound<'_, PyAny>,
        pixels_per_cube: f64,
        inv_scale: f64,
        min_dist: f64,
        memory_limit_mb: i32,
        bisection_iters: i32,
        bisection_tol: f64,
        enclosed: bool,
        simplify_occluded: bool,
        visible_relax_iter: i32,
        coarse_count: i32,
    ) -> PyResult<Self> {
        // Load the shared library
        let lib = CoreLib::load(lib_path).map_err(PyErr::from)?;

        // Parse cameras
        if cameras.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cameras must be a 4-tuple: (cam_poses, Ks, Hs, Ws)",
            ));
        }
        let cam_poses_py = cameras.get_item(0)?;
        let ks_py = cameras.get_item(1)?;
        let hs_py = cameras.get_item(2)?;
        let ws_py = cameras.get_item(3)?;

        let cam_poses_list: Vec<Vec<f64>> = cam_poses_py
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("cam_poses must be list of 4x4 arrays"))?;
        let ks_list: Vec<Vec<f64>> = ks_py
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Ks must be list of 3x3 arrays"))?;
        let hs_list: Vec<f64> = hs_py
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Hs must be list of numbers"))?;
        let ws_list: Vec<f64> = ws_py
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Ws must be list of numbers"))?;

        let n_cams = hs_list.len();
        if cam_poses_list.len() != n_cams || ks_list.len() != n_cams || ws_list.len() != n_cams {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Camera arrays must all have the same length",
            ));
        }

        let mut cam_poses_flat: Vec<f64> = Vec::with_capacity(n_cams * 16);
        for pose in &cam_poses_list {
            if pose.len() != 16 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each cam_pose must be a flattened 4x4 matrix (16 elements)",
                ));
            }
            cam_poses_flat.extend_from_slice(pose);
        }
        let mut ks_flat: Vec<f64> = Vec::with_capacity(n_cams * 9);
        for k in &ks_list {
            if k.len() != 9 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each K must be a flattened 3x3 matrix (9 elements)",
                ));
            }
            ks_flat.extend_from_slice(k);
        }

        let cameras_data =
            pack_cameras(&cam_poses_flat, &ks_flat, &hs_list, &ws_list).map_err(PyErr::from)?;

        // Parse bounds
        let bounds_vec: Vec<f64> = bounds
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bounds must be list of 6 floats"))?;
        let (b_min, b_max, center, size) = validate_bounds(&bounds_vec).map_err(PyErr::from)?;

        // Detect hardware capabilities for get_capabilities()
        let (supports_cuda, supports_mps) = detect_torch_capabilities(py);

        let params = MesherParams {
            cameras_data,
            n_cams: n_cams as i32,
            center,
            size,
            bounds_min: b_min,
            bounds_max: b_max,
            pixels_per_cube,
            inv_scale,
            min_dist,
            memory_limit_mb,
            bisection_iters,
            bisection_tol,
            enclosed,
            simplify_occluded,
            visible_relax_iter,
            coarse_count,
        };

        Ok(Backend {
            lib: Arc::new(lib),
            params,
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_cuda,
            supports_mps,
        })
    }

    /// Return capability flags consumed by RustOcMesher.capabilities().
    fn get_capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("supports_cpu", true)?;
        d.set_item("supports_cuda", self.supports_cuda)?;
        d.set_item("supports_mps", self.supports_mps)?;
        d.set_item("preferred_dtype", "float32")?;
        d.set_item("max_batch", py.None())?;
        d.set_item("supports_async", false)?;
        d.set_item("default_stream_policy", "sync")?;
        d.set_item("version", self.version.as_str())?;
        Ok(d)
    }

    /// Run the meshing pipeline and return ``(meshes, in_view_tags)``.
    ///
    /// Parameters
    /// ----------
    /// sdf_kernels : list[callable]
    ///     One callable per SDF element.  Each callable receives an ``(N, 3)``
    ///     float64 numpy array and must return an ``(N,)`` float32/float64 array
    ///     (or a CPU torch.Tensor / DLPack-capable object).
    fn extract_meshes<'py>(
        &self,
        py: Python<'py>,
        sdf_kernels: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let kernels: Vec<Py<PyAny>> = sdf_kernels
            .iter()
            .map(|k| k.unbind())
            .collect();

        if kernels.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sdf_kernels must be a non-empty list",
            ));
        }
        for (i, k) in kernels.iter().enumerate() {
            if !k.bind(py).is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "sdf_kernels[{i}] is not callable"
                )));
            }
        }

        let mesh_data_list = run_meshing_pipeline(py, &self.lib, &self.params, &kernels)?;

        let meshes_list = PyList::empty_bound(py);
        let tags_list = PyList::empty_bound(py);

        for mesh_data in mesh_data_list {
            let (mesh_obj, tag_np) = ocmesher_core::mesh_data_to_python(py, mesh_data)?;
            meshes_list.append(mesh_obj)?;
            tags_list.append(tag_np)?;
        }

        let result = PyTuple::new_bound(py, [meshes_list.into_any(), tags_list.into_any()]);
        Ok(result)
    }

    /// Run the meshing pipeline with a built-in Rust-native sphere SDF.
    #[pyo3(signature = (radius = 1.0, center = None))]
    fn extract_native_sphere<'py>(
        &self,
        py: Python<'py>,
        radius: f64,
        center: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let center_arr = match center {
            Some(values) => {
                if values.len() != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "center must contain exactly 3 values",
                    ));
                }
                [values[0], values[1], values[2]]
            }
            None => [0.0, 0.0, 0.0],
        };
        let kernels = vec![Box::new(SphereKernel::new(center_arr, radius)) as _];
        let mesh_data_list = run_meshing_pipeline_native(&self.lib, &self.params, &kernels)
            .map_err(PyErr::from)?;

        let meshes_list = PyList::empty_bound(py);
        let tags_list = PyList::empty_bound(py);

        for mesh_data in mesh_data_list {
            let (mesh_obj, tag_np) = ocmesher_core::mesh_data_to_python(py, mesh_data)?;
            meshes_list.append(mesh_obj)?;
            tags_list.append(tag_np)?;
        }

        Ok(PyTuple::new_bound(py, [meshes_list.into_any(), tags_list.into_any()]))
    }

    /// Run the meshing pipeline with a `tch`-backed sphere SDF.
    #[cfg(feature = "tch-kernels")]
    #[pyo3(signature = (radius = 1.0, center = None, device = None))]
    fn extract_tch_sphere<'py>(
        &self,
        py: Python<'py>,
        radius: f64,
        center: Option<Vec<f64>>,
        device: Option<String>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let center_arr = match center {
            Some(values) => {
                if values.len() != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "center must contain exactly 3 values",
                    ));
                }
                [values[0] as f32, values[1] as f32, values[2] as f32]
            }
            None => [0.0, 0.0, 0.0],
        };
        let tch_device = match device.as_deref() {
            None | Some("cpu") => Device::Cpu,
            Some("mps") => Device::Mps,
            Some(value) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported tch device: {value}"
                )))
            }
        };
        let kernels = vec![Box::new(TchSphereKernel::new(center_arr, radius as f32, tch_device)) as _];
        let mesh_data_list = run_meshing_pipeline_native(&self.lib, &self.params, &kernels)
            .map_err(PyErr::from)?;

        let meshes_list = PyList::empty_bound(py);
        let tags_list = PyList::empty_bound(py);

        for mesh_data in mesh_data_list {
            let (mesh_obj, tag_np) = ocmesher_core::mesh_data_to_python(py, mesh_data)?;
            meshes_list.append(mesh_obj)?;
            tags_list.append(tag_np)?;
        }

        Ok(PyTuple::new_bound(py, [meshes_list.into_any(), tags_list.into_any()]))
    }

    /// Convenience: same as ``get_capabilities()`` for protocol compatibility.
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.get_capabilities(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ocmesher_rust.Backend(n_cams={}, version={:?})",
            self.params.n_cams, self.version
        )
    }

    #[getter]
    fn version(&self) -> &str {
        &self.version
    }
}

// ---------------------------------------------------------------------------
// Hardware detection helper (optional torch import)
// ---------------------------------------------------------------------------

fn detect_torch_capabilities(py: Python<'_>) -> (bool, bool) {
    let Ok(sys) = py.import_bound("sys") else {
        return (false, false);
    };
    let Ok(modules) = sys.getattr("modules") else {
        return (false, false);
    };
    let Ok(torch) = modules.get_item("torch") else {
        return (false, false);
    };
    if torch.is_none() {
        return (false, false);
    }
    let cuda = torch
        .getattr("cuda")
        .and_then(|c| c.call_method0("is_available"))
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let mps = torch
        .getattr("backends")
        .and_then(|b| b.getattr("mps"))
        .and_then(|m| m.call_method0("is_available"))
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    (cuda, mps)
}

// ---------------------------------------------------------------------------
// find_core_so: helper exported to Python to locate the prebuilt core.so
// ---------------------------------------------------------------------------

/// Return the path to the OcMesher ``core.so`` found via the ``ocmesher`` package.
///
/// Raises ``RuntimeError`` if the library cannot be located.
#[pyfunction]
fn find_core_so(py: Python<'_>) -> PyResult<String> {
    let ocmesher = py.import_bound("ocmesher")?;
    let pkg_file: String = ocmesher
        .getattr("__file__")?
        .extract()?;
    let pkg_dir = std::path::Path::new(&pkg_file)
        .parent()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("cannot resolve package dir"))?;
    let so_path = pkg_dir.join("lib").join("core.so");
    if so_path.exists() {
        Ok(so_path.to_string_lossy().into_owned())
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "core.so not found at {so_path:?}; run 'bash install.sh' first"
        )))
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// PyO3 extension module for the OcMesher Rust backend.
#[pymodule]
fn ocmesher_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Backend>()?;
    m.add_function(wrap_pyfunction!(find_core_so, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
