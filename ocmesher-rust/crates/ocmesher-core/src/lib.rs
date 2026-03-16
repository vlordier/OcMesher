// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

//! Rust core for the OcMesher Rust backend.
//!
//! Provides:
//! - [`CoreLib`]: runtime loader for `core.so` via `libloading`.
//! - [`pack_cameras`]: packs Python camera tuples into the flat array expected by C++.
//! - [`MeshData`]: plain-data result struct (vertices + faces + in-view tags).
//! - [`run_meshing_pipeline`]: full coarse-to-fine pipeline matching `OcMesher.__call__`.
//!
//! # Global state
//!
//! The C++ `core.so` uses global mutable variables.  A process-wide [`Mutex`] is
//! acquired for the entire duration of [`run_meshing_pipeline`] so that concurrent
//! Python calls cannot interleave.

use std::ffi::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

struct DLPackTensorContext {
    storage: Vec<f64>,
    shape: [i64; 2],
}

unsafe extern "C" fn dl_managed_tensor_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    let managed = Box::from_raw(managed);
    if !managed.manager_ctx.is_null() {
        drop(Box::from_raw(
            managed.manager_ctx.cast::<DLPackTensorContext>(),
        ));
    }
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(capsule, DLPACK_CAPSULE_NAME.as_ptr().cast()) == 0 {
        return;
    }
    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr().cast());
    if ptr.is_null() {
        return;
    }
    let managed = ptr.cast::<DLManagedTensor>();
    if let Some(deleter) = (*managed).deleter {
        deleter(managed);
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("failed to load core.so: {0}")]
    Load(#[from] libloading::Error),
    #[error("camera validation failed: {0}")]
    Camera(String),
    #[error("bounds validation failed: {0}")]
    Bounds(String),
    #[error("SDF kernel call failed: {0}")]
    Sdf(String),
    #[error("ndarray shape error: {0}")]
    Shape(String),
}

impl From<CoreError> for PyErr {
    fn from(e: CoreError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Process-wide serialisation lock (C++ global state)
// ---------------------------------------------------------------------------

static CORE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn core_lock() -> &'static Mutex<()> {
    CORE_LOCK.get_or_init(|| Mutex::new(()))
}

// ---------------------------------------------------------------------------
// Function pointer type aliases matching C++ extern "C" ABI
// ---------------------------------------------------------------------------

type FnRunCoarse = unsafe extern "C" fn(
    *mut f64,
    f64,
    c_int,
    *mut f64,
    f64,
    f64,
    f64,
    c_int,
    c_int,
    c_int,
) -> c_int;
type FnFineGroup = unsafe extern "C" fn() -> c_int;
type FnFineIteration = unsafe extern "C" fn(*mut f32) -> c_int;
type FnFineIterationOutput = unsafe extern "C" fn(*mut f64);
type FnVisFilter = unsafe extern "C" fn(bool, c_int) -> c_int;
type FnFinalIteration = unsafe extern "C" fn(*mut c_int) -> c_int;
type FnFinalIterationOccluded = unsafe extern "C" fn(*mut c_int) -> c_int;
type FnFinalIteration2 = unsafe extern "C" fn(*mut f64);
type FnFinalIteration3 = unsafe extern "C" fn(*mut f32) -> c_int;
type FnFinalIteration3Occluded = unsafe extern "C" fn(*mut f32);
type FnFinalRemaining = unsafe extern "C" fn(*mut c_int);
type FnGetVertsCenter = unsafe extern "C" fn(c_int, *mut f64);
type FnUpdateVerts = unsafe extern "C" fn(c_int, *const f32, *const f32, *mut f64);
type FnGetLrVerts = unsafe extern "C" fn(c_int, *mut f64, *mut f64);
type FnFinalizeVerts = unsafe extern "C" fn(c_int, *const f32, *const f32, *mut f64);
type FnConstructFaces = unsafe extern "C" fn(c_int, *mut f64, *mut c_int);
type FnGetExtraVertsCenter = unsafe extern "C" fn(*mut f64, *mut f64);
type FnUpdateExtraVerts =
    unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f64, *mut f64);
type FnGetLrExtraVerts = unsafe extern "C" fn(*mut f64, *mut f64, *mut f64, *mut f64);
type FnFinalizeExtraVerts =
    unsafe extern "C" fn(*const f32, *const f32, *mut f64, *const f32, *const f32, *mut f64);
type FnGetFaces = unsafe extern "C" fn(*mut c_int);
type FnGetInViewTag = unsafe extern "C" fn(c_int, *mut bool);

// ---------------------------------------------------------------------------
// CoreLib: dynamic loader
// ---------------------------------------------------------------------------

/// Handle to a loaded `core.so`, containing typed function pointers.
///
/// The underlying `Library` is kept alive by this struct.
pub struct CoreLib {
    // Keep the library alive: symbol lifetimes are tied to `_lib`.
    _lib: Library,

    // Cached typed function pointers.
    run_coarse: FnRunCoarse,
    fine_group: FnFineGroup,
    fine_iteration: FnFineIteration,
    fine_iteration_output: FnFineIterationOutput,
    vis_filter: FnVisFilter,
    final_iteration: FnFinalIteration,
    final_iteration_occluded: FnFinalIterationOccluded,
    final_iteration2: FnFinalIteration2,
    final_iteration3: FnFinalIteration3,
    final_iteration3_occluded: FnFinalIteration3Occluded,
    final_remaining: FnFinalRemaining,
    get_verts_center: FnGetVertsCenter,
    update_verts: FnUpdateVerts,
    get_lr_verts: FnGetLrVerts,
    finalize_verts: FnFinalizeVerts,
    construct_faces: FnConstructFaces,
    get_extra_verts_center: FnGetExtraVertsCenter,
    update_extra_verts: FnUpdateExtraVerts,
    get_lr_extra_verts: FnGetLrExtraVerts,
    finalize_extra_verts: FnFinalizeExtraVerts,
    get_faces: FnGetFaces,
    get_in_view_tag: FnGetInViewTag,
}

// SAFETY: CoreLib is only callable while holding CORE_LOCK, ensuring
// exclusive access to the C++ global state.
unsafe impl Send for CoreLib {}
unsafe impl Sync for CoreLib {}

macro_rules! load_sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let sym: Symbol<$ty> = $lib.get(concat!($name, "\0").as_bytes())?;
        *sym
    }};
}

impl CoreLib {
    /// Dynamically load `core.so` from `path` and resolve all required symbols.
    pub fn load(path: &str) -> Result<Self, CoreError> {
        // SAFETY: we're loading a trusted library built from the OcMesher source.
        let lib = unsafe { Library::new(path) }?;
        unsafe {
            Ok(CoreLib {
                run_coarse: load_sym!(lib, "run_coarse", FnRunCoarse),
                fine_group: load_sym!(lib, "fine_group", FnFineGroup),
                fine_iteration: load_sym!(lib, "fine_iteration", FnFineIteration),
                fine_iteration_output: load_sym!(lib, "fine_iteration_output", FnFineIterationOutput),
                vis_filter: load_sym!(lib, "vis_filter", FnVisFilter),
                final_iteration: load_sym!(lib, "final_iteration", FnFinalIteration),
                final_iteration_occluded: load_sym!(
                    lib,
                    "final_iteration_occluded",
                    FnFinalIterationOccluded
                ),
                final_iteration2: load_sym!(lib, "final_iteration2", FnFinalIteration2),
                final_iteration3: load_sym!(lib, "final_iteration3", FnFinalIteration3),
                final_iteration3_occluded: load_sym!(
                    lib,
                    "final_iteration3_occluded",
                    FnFinalIteration3Occluded
                ),
                final_remaining: load_sym!(lib, "final_remaining", FnFinalRemaining),
                get_verts_center: load_sym!(lib, "get_verts_center", FnGetVertsCenter),
                update_verts: load_sym!(lib, "update_verts", FnUpdateVerts),
                get_lr_verts: load_sym!(lib, "get_lr_verts", FnGetLrVerts),
                finalize_verts: load_sym!(lib, "finalize_verts", FnFinalizeVerts),
                construct_faces: load_sym!(lib, "construct_faces", FnConstructFaces),
                get_extra_verts_center: load_sym!(
                    lib,
                    "get_extra_verts_center",
                    FnGetExtraVertsCenter
                ),
                update_extra_verts: load_sym!(lib, "update_extra_verts", FnUpdateExtraVerts),
                get_lr_extra_verts: load_sym!(lib, "get_lr_extra_verts", FnGetLrExtraVerts),
                finalize_extra_verts: load_sym!(lib, "finalize_extra_verts", FnFinalizeExtraVerts),
                get_faces: load_sym!(lib, "get_faces", FnGetFaces),
                get_in_view_tag: load_sym!(lib, "get_in_view_tag", FnGetInViewTag),
                _lib: lib,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Camera packing (matches core.py __init__)
// ---------------------------------------------------------------------------

/// Camera data stride: [inv_pose_3x4 (12) | K_3x3 (9) | H (1) | W (1)] per camera.
pub const CAMERA_DATA_STRIDE: usize = 23;

/// Invert an SE(3) camera pose stored as 4×4 row-major f64 array.
///
/// For rigid-body transforms `P = [R | t; 0 | 1]`:
/// `P⁻¹ = [Rᵀ | −Rᵀt; 0 | 1]`
fn invert_pose(pose: &[f64]) -> [f64; 16] {
    assert_eq!(pose.len(), 16);
    let r = |i: usize, j: usize| pose[i * 4 + j];
    let t = |i: usize| pose[i * 4 + 3];
    let mut inv = [0.0f64; 16];
    // Upper-left 3×3: Rᵀ
    for i in 0..3 {
        for j in 0..3 {
            inv[i * 4 + j] = r(j, i);
        }
    }
    // Upper-right 3×1: −Rᵀ t
    for i in 0..3 {
        let mut dot = 0.0f64;
        for j in 0..3 {
            dot += r(j, i) * t(j);
        }
        inv[i * 4 + 3] = -dot;
    }
    // Bottom row: [0, 0, 0, 1]
    inv[15] = 1.0;
    inv
}

/// Pack camera data into the flat buffer expected by `run_coarse`.
///
/// Returns `(packed_data, n_cams)` where packed data is `n_cams × CAMERA_DATA_STRIDE` f64 values.
///
/// `cam_poses_flat` – `n_cams × 16` row-major f64 values for each 4×4 pose.
/// `ks_flat`        – `n_cams × 9`  row-major f64 values for each 3×3 K matrix.
/// `hs`             – `n_cams` image heights (f64).
/// `ws`             – `n_cams` image widths  (f64).
pub fn pack_cameras(
    cam_poses_flat: &[f64],
    ks_flat: &[f64],
    hs: &[f64],
    ws: &[f64],
) -> Result<Vec<f64>, CoreError> {
    let n = hs.len();
    if cam_poses_flat.len() != n * 16 || ks_flat.len() != n * 9 || ws.len() != n {
        return Err(CoreError::Camera(format!(
            "Inconsistent camera array lengths: poses={}, Ks={}, Hs={}, Ws={}",
            cam_poses_flat.len(),
            ks_flat.len(),
            n,
            ws.len()
        )));
    }

    let mut packed = vec![0.0f64; n * CAMERA_DATA_STRIDE];
    for cam in 0..n {
        let base = cam * CAMERA_DATA_STRIDE;
        let inv = invert_pose(&cam_poses_flat[cam * 16..(cam + 1) * 16]);
        // First 12: first 3 rows of 4×4 inverse (3×4 submatrix)
        for row in 0..3 {
            for col in 0..4 {
                packed[base + row * 4 + col] = inv[row * 4 + col];
            }
        }
        // Next 9: K matrix (3×3)
        packed[base + 12..base + 21].copy_from_slice(&ks_flat[cam * 9..(cam + 1) * 9]);
        // H and W
        packed[base + 21] = hs[cam];
        packed[base + 22] = ws[cam];
    }
    Ok(packed)
}

/// Validate and extract scalar bounds from a 6-element Python sequence.
pub fn validate_bounds(bounds: &[f64]) -> Result<([f64; 3], [f64; 3], [f64; 3], f64), CoreError> {
    if bounds.len() != 6 {
        return Err(CoreError::Bounds(format!(
            "bounds must have 6 elements, got {}",
            bounds.len()
        )));
    }
    for &v in bounds {
        if !v.is_finite() {
            return Err(CoreError::Bounds("bounds contain non-finite value".into()));
        }
    }
    for axis in 0..3 {
        if bounds[axis * 2] >= bounds[axis * 2 + 1] {
            return Err(CoreError::Bounds(format!(
                "bounds min >= max on axis {}",
                axis
            )));
        }
    }
    let b_min = [bounds[0], bounds[2], bounds[4]];
    let b_max = [bounds[1], bounds[3], bounds[5]];
    let center = [
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    ];
    let size = (b_max[0] - b_min[0])
        .max(b_max[1] - b_min[1])
        .max(b_max[2] - b_min[2])
        * 1.1;
    Ok((b_min, b_max, center, size))
}

// ---------------------------------------------------------------------------
// SDF evaluation helpers
// ---------------------------------------------------------------------------

fn extract_sdf_value(py: Python<'_>, raw: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let raw_bound = raw.bind(py);
    if let Ok(dict) = raw_bound.downcast::<PyDict>() {
        if let Ok(Some(v)) = dict.get_item("sdf") {
            Ok(v.unbind())
        } else if let Ok(Some(v)) = dict.get_item("SDF") {
            Ok(v.unbind())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(
                "SDF output dict must contain 'sdf' or 'SDF'",
            ))
        }
    } else {
        Ok(raw.clone_ref(py))
    }
}

fn extract_sdf_output(py: Python<'_>, raw: Py<PyAny>) -> PyResult<Vec<f32>> {
    use numpy::PyReadonlyArray1;

    let raw = extract_sdf_value(py, raw)?;

    if let Ok(arr) = raw.extract::<PyReadonlyArray1<f32>>(py) {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
    }

    if let Ok(arr) = raw.extract::<PyReadonlyArray1<f64>>(py) {
        return arr
            .as_slice()
            .map(|s| s.iter().map(|&v| v as f32).collect())
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
    }

    if let Ok(detached) = raw.bind(py).call_method0("detach") {
        if let Ok(cpu_tensor) = detached.call_method1("to", ("cpu",)) {
            if let Ok(numpy_obj) = cpu_tensor.call_method0("numpy") {
                if let Ok(arr) = numpy_obj.extract::<PyReadonlyArray1<f32>>() {
                    return arr
                        .as_slice()
                        .map(|s| s.to_vec())
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
                }
                if let Ok(arr) = numpy_obj.extract::<PyReadonlyArray1<f64>>() {
                    return arr
                        .as_slice()
                        .map(|s| s.iter().map(|&v| v as f32).collect())
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
                }
            }
        }
    }

    if let Ok(numpy_obj) = raw.bind(py).call_method0("numpy") {
        if let Ok(arr) = numpy_obj.extract::<PyReadonlyArray1<f32>>() {
            return arr
                .as_slice()
                .map(|s| s.to_vec())
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
        }
        if let Ok(arr) = numpy_obj.extract::<PyReadonlyArray1<f64>>() {
            return arr
                .as_slice()
                .map(|s| s.iter().map(|&v| v as f32).collect())
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
        }
    }

    if let Ok(cap) = raw.bind(py).call_method0("__dlpack__") {
        let np = py.import_bound("numpy")?;
        let from_dlpack = np.getattr("from_dlpack")?;
        let arr_obj = from_dlpack.call1((cap,))?;
        if let Ok(arr) = arr_obj.extract::<PyReadonlyArray1<f32>>() {
            return arr
                .as_slice()
                .map(|s| s.to_vec())
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "SDF kernel return value must be a float32/float64 numpy array, \
         a CPU torch.Tensor, or a DLPack-capable object",
    ))
}

fn build_torch_xyz_input_from_numpy<'py>(
    py: Python<'py>,
    xyz_np: &Bound<'py, PyAny>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import_bound("torch")?;
    let from_numpy = torch.getattr("from_numpy")?;
    let xyz_t = from_numpy.call1((xyz_np.clone(),))?;
    let dtype_name = torch_eval_dtype.unwrap_or("float32");
    let dtype_obj = if dtype_name == "float64" {
        torch.getattr("float64")?
    } else {
        torch.getattr("float32")?
    };
    if let Some(device) = torch_eval_device {
        xyz_t.call_method1("to", (device, dtype_obj))
    } else {
        xyz_t.call_method1("to", (dtype_obj,))
    }
}

fn build_torch_xyz_input_dlpack<'py>(
    py: Python<'py>,
    xyz: &[f64],
    n_pts: usize,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let ctx = Box::new(DLPackTensorContext {
        storage: xyz.to_vec(),
        shape: [n_pts as i64, 3],
    });
    let ctx_ptr = Box::into_raw(ctx);
    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: unsafe { (*ctx_ptr).storage.as_mut_ptr().cast::<c_void>() },
            device: DLDevice {
                device_type: 1,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: 2,
                bits: 64,
                lanes: 1,
            },
            shape: unsafe { (*ctx_ptr).shape.as_mut_ptr() },
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr.cast::<c_void>(),
        deleter: Some(dl_managed_tensor_deleter),
    });
    let managed_ptr = Box::into_raw(managed);

    let capsule = unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            pyo3::ffi::PyCapsule_New(
                managed_ptr.cast::<c_void>(),
                DLPACK_CAPSULE_NAME.as_ptr().cast(),
                Some(dlpack_capsule_destructor),
            ),
        )?
    };

    let torch = py.import_bound("torch")?;
    let from_dlpack = torch.getattr("utils")?.getattr("dlpack")?.getattr("from_dlpack")?;
    let xyz_t = from_dlpack.call1((capsule,))?;
    let dtype_obj = build_torch_dtype(&torch.into_any(), torch_eval_dtype)?;
    let xyz_t = if let Some(device) = torch_eval_device {
        xyz_t.call_method1("to", (device, dtype_obj))?
    } else if torch_eval_dtype.unwrap_or("float32") == "float64" {
        xyz_t
    } else {
        xyz_t.call_method1("to", (dtype_obj,))?
    };

    Ok(xyz_t)
}

fn build_torch_dtype<'py>(
    torch: &Bound<'py, PyAny>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype_name = torch_eval_dtype.unwrap_or("float32");
    if dtype_name == "float64" {
        torch.getattr("float64")
    } else {
        torch.getattr("float32")
    }
}

fn normalize_torch_output<'py>(
    py: Python<'py>,
    value: Py<PyAny>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import_bound("torch")?;
    let as_tensor = torch.getattr("as_tensor")?;
    let dtype_obj = build_torch_dtype(&torch.into_any(), torch_eval_dtype)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", dtype_obj)?;
    if let Some(device) = torch_eval_device {
        kwargs.set_item("device", device)?;
        as_tensor.call((value.bind(py),), Some(&kwargs))
    } else {
        as_tensor.call((value.bind(py),), Some(&kwargs))
    }
}

fn eval_kernel_raw_from_inputs<'py>(
    py: Python<'py>,
    kernel: &Py<PyAny>,
    xyz_np: Option<&Bound<'py, PyAny>>,
    xyz_torch: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let kernel_bound = kernel.bind(py);
    if let Some(xyz_t) = xyz_torch {
        if let Ok(eval_batch_torch) = kernel_bound.getattr("evaluate_batch_torch") {
            if eval_batch_torch.is_callable() {
                return Ok(eval_batch_torch.call1((xyz_t.clone(),))?.unbind());
            }
        }
    }

    if let Ok(eval_batch) = kernel_bound.getattr("evaluate_batch") {
        if eval_batch.is_callable() {
            if let Some(xyz_np) = xyz_np {
                return Ok(eval_batch.call1((xyz_np.clone(),))?.unbind());
            }
        }
    }

    if let Some(xyz_np) = xyz_np {
        return Ok(kernel_bound.call1((xyz_np.clone(),))?.unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "kernel requires numpy input but no numpy query matrix was provided",
    ))
}

fn eval_kernel_from_inputs<'py>(
    py: Python<'py>,
    kernel: &Py<PyAny>,
    xyz_np: Option<&Bound<'py, PyAny>>,
    xyz_torch: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<f32>> {
    let raw = eval_kernel_raw_from_inputs(py, kernel, xyz_np, xyz_torch)?;
    extract_sdf_output(py, raw)
}

fn all_kernels_support_torch_eval(py: Python<'_>, kernels: &[Py<PyAny>]) -> bool {
    kernels.iter().all(|kernel| {
        kernel
            .bind(py)
            .getattr("evaluate_batch_torch")
            .map(|attr| attr.is_callable())
            .unwrap_or(false)
    })
}

fn shared_torch_bundle<'py>(py: Python<'py>, kernels: &[Py<PyAny>]) -> Option<Bound<'py, PyAny>> {
    if kernels.len() <= 1 {
        return None;
    }
    let mut shared_bundle: Option<Py<PyAny>> = None;
    for kernel in kernels {
        let Ok(bundle) = kernel.bind(py).getattr("_ocmesher_torch_bundle") else {
            return None;
        };
        match &shared_bundle {
            None => shared_bundle = Some(bundle.unbind()),
            Some(existing) => {
                if !bundle.is(existing.bind(py)) {
                    return None;
                }
            }
        }
    }
    shared_bundle.map(|bundle| bundle.into_bound(py))
}

fn eval_bundle_raw_from_inputs<'py>(
    _py: Python<'py>,
    bundle: &Bound<'py, PyAny>,
    xyz_np: Option<&Bound<'py, PyAny>>,
    xyz_torch: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if let Some(xyz_t) = xyz_torch {
        if let Ok(eval_many_torch) = bundle.getattr("evaluate_many_torch") {
            if eval_many_torch.is_callable() {
                return Ok(eval_many_torch.call1((xyz_t.clone(),))?.unbind());
            }
        }
    }
    if let Ok(eval_many) = bundle.getattr("evaluate_many") {
        if eval_many.is_callable() {
            if let Some(xyz_np) = xyz_np {
                return Ok(eval_many.call1((xyz_np.clone(),))?.unbind());
            }
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "shared torch bundle must provide evaluate_many_torch or provide evaluate_many with numpy input",
    ))
}

/// Call a single Python SDF kernel with `xyz` (n_pts × 3 f64) and return f32 values.
///
/// Prefers `kernel.evaluate_batch(xyz)` when available, otherwise falls back to
/// `kernel(xyz)`. Accepts numpy float32/float64 arrays, dict outputs containing
/// `{"sdf": ...}` or `{"SDF": ...}`, or objects that expose `.numpy()`.
fn eval_kernel_py_once(
    py: Python<'_>,
    kernel: &Py<PyAny>,
    xyz: &[f64],
    n_pts: usize,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Vec<f32>> {
    use ndarray::Array2;
    use numpy::IntoPyArray;

    let xyz_arr = Array2::from_shape_vec((n_pts, 3), xyz.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let xyz_np = xyz_arr.into_pyarray_bound(py).into_any();
    let xyz_torch = if torch_eval_device.is_some() {
        Some(build_torch_xyz_input_from_numpy(
            py,
            &xyz_np,
            torch_eval_device,
            torch_eval_dtype,
        )?)
    } else {
        None
    };
    eval_kernel_from_inputs(py, kernel, Some(&xyz_np), xyz_torch.as_ref())
}

fn eval_kernel_py(
    py: Python<'_>,
    kernel: &Py<PyAny>,
    xyz: &[f64],
    n_pts: usize,
    sdf_batch_size: Option<usize>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
) -> PyResult<Vec<f32>> {
    let Some(batch_size) = sdf_batch_size else {
        return eval_kernel_py_once(py, kernel, xyz, n_pts, torch_eval_device, torch_eval_dtype);
    };

    if batch_size == 0 || n_pts <= batch_size {
        return eval_kernel_py_once(py, kernel, xyz, n_pts, torch_eval_device, torch_eval_dtype);
    }

    let mut out = Vec::with_capacity(n_pts);
    for start in (0..n_pts).step_by(batch_size) {
        let end = (start + batch_size).min(n_pts);
        let chunk = &xyz[start * 3..end * 3];
        let chunk_vals = eval_kernel_py_once(
            py,
            kernel,
            chunk,
            end - start,
            torch_eval_device,
            torch_eval_dtype,
        )?;
        out.extend(chunk_vals);
    }
    Ok(out)
}

/// Evaluate all `kernels` at `xyz` (n_pts × 3 f64) and return a flat f32 array of shape
/// `(n_pts × n_kernels)` in row-major order: `result[i * n_kernels + k]` = kernel k value at
/// point i.
///
/// If `enclosed`, points outside `[b_min, b_max]` are clamped to `+1.0` (exterior).
fn eval_sdf_full(
    py: Python<'_>,
    kernels: &[Py<PyAny>],
    xyz: &[f64],
    n_pts: usize,
    n_kernels: usize,
    sdf_batch_size: Option<usize>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<Vec<f32>> {
    let mut result = vec![0.0f32; n_pts * n_kernels];

    let use_fused_torch = torch_eval_device.is_some() && n_kernels > 1 && all_kernels_support_torch_eval(py, kernels);
    let torch_bundle = if use_fused_torch { shared_torch_bundle(py, kernels) } else { None };

    if use_fused_torch {
        let chunk_size = sdf_batch_size.unwrap_or(n_pts).max(1);
        for start in (0..n_pts).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_pts);
            let chunk_n = end - start;
            let chunk_xyz = &xyz[start * 3..end * 3];
            let xyz_torch = build_torch_xyz_input_dlpack(
                py,
                chunk_xyz,
                chunk_n,
                torch_eval_device,
                torch_eval_dtype,
            )?;

            if let Some(bundle) = &torch_bundle {
                let raw = eval_bundle_raw_from_inputs(py, bundle, None, Some(&xyz_torch))?;
                let value = extract_sdf_value(py, raw)?;
                let tensor = normalize_torch_output(py, value, torch_eval_device, torch_eval_dtype)?;
                let chunk_vals = extract_sdf_output(py, tensor.unbind())?;
                if chunk_vals.len() != chunk_n * n_kernels {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "fused torch bundle returned {} values for {}x{} query matrix",
                        chunk_vals.len(),
                        chunk_n,
                        n_kernels,
                    )));
                }
                for local_i in 0..chunk_n {
                    let global_i = start + local_i;
                    let src = &chunk_vals[local_i * n_kernels..(local_i + 1) * n_kernels];
                    let dst = &mut result[global_i * n_kernels..(global_i + 1) * n_kernels];
                    dst.copy_from_slice(src);
                }
                continue;
            }

            for (k_idx, kernel) in kernels.iter().enumerate() {
                let sdf = eval_kernel_from_inputs(py, kernel, None, Some(&xyz_torch))?;
                if sdf.len() != chunk_n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "kernels[{k_idx}] returned {} values for {chunk_n} query points",
                        sdf.len()
                    )));
                }
                for (local_i, v) in sdf.iter().enumerate() {
                    let global_i = start + local_i;
                    result[global_i * n_kernels + k_idx] = *v;
                }
            }
        }
    } else {

        for (k_idx, kernel) in kernels.iter().enumerate() {
            let sdf = eval_kernel_py(
                py,
                kernel,
                xyz,
                n_pts,
                sdf_batch_size,
                torch_eval_device,
                torch_eval_dtype,
            )?;
            if sdf.len() != n_pts {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "kernels[{k_idx}] returned {} values for {n_pts} query points",
                    sdf.len()
                )));
            }
            for (i, v) in sdf.iter().enumerate() {
                result[i * n_kernels + k_idx] = *v;
            }
        }
    }

    // Apply bounds masking in-place
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

/// Like [`eval_sdf_full`] but returns a single f32 per point: the minimum across all kernels.
/// Used during the coarse pass where `fine_iteration` only cares about the sign.
fn eval_sdf_min(
    py: Python<'_>,
    kernels: &[Py<PyAny>],
    xyz: &[f64],
    n_pts: usize,
    sdf_batch_size: Option<usize>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<Vec<f32>> {
    let n_kernels = kernels.len();
    let use_fused_torch = torch_eval_device.is_some() && n_kernels > 1 && all_kernels_support_torch_eval(py, kernels);
    let torch_bundle = if use_fused_torch { shared_torch_bundle(py, kernels) } else { None };

    if use_fused_torch {
        let torch = py.import_bound("torch")?;
        let stack = torch.getattr("stack")?;
        let chunk_size = sdf_batch_size.unwrap_or(n_pts).max(1);
        let mut min_sdf = vec![f32::INFINITY; n_pts];

        for start in (0..n_pts).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_pts);
            let chunk_n = end - start;
            let chunk_xyz = &xyz[start * 3..end * 3];
            let xyz_torch = build_torch_xyz_input_dlpack(
                py,
                chunk_xyz,
                chunk_n,
                torch_eval_device,
                torch_eval_dtype,
            )?;

            if let Some(bundle) = &torch_bundle {
                let raw = eval_bundle_raw_from_inputs(py, bundle, None, Some(&xyz_torch))?;
                let value = extract_sdf_value(py, raw)?;
                let tensor = normalize_torch_output(py, value, torch_eval_device, torch_eval_dtype)?;
                let reduced = tensor.call_method1("amin", (1,))?;
                let chunk_vals = extract_sdf_output(py, reduced.unbind())?;
                if chunk_vals.len() != chunk_n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "fused torch bundle reduction returned {} values for {chunk_n} query points",
                        chunk_vals.len()
                    )));
                }
                min_sdf[start..end].copy_from_slice(&chunk_vals);
                continue;
            }
            let outputs = PyList::empty_bound(py);

            for kernel in kernels {
                let raw = eval_kernel_raw_from_inputs(py, kernel, None, Some(&xyz_torch))?;
                let value = extract_sdf_value(py, raw)?;
                let tensor = normalize_torch_output(py, value, torch_eval_device, torch_eval_dtype)?;
                outputs.append(tensor)?;
            }

            let stacked = stack.call1((outputs, 1))?;
            let reduced = stacked.call_method1("amin", (1,))?;
            let chunk_vals = extract_sdf_output(py, reduced.unbind())?;
            if chunk_vals.len() != chunk_n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "fused torch reduction returned {} values for {chunk_n} query points",
                    chunk_vals.len()
                )));
            }
            min_sdf[start..end].copy_from_slice(&chunk_vals);
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
                    min_sdf[i] = 1.0;
                }
            }
        }

        return Ok(min_sdf);
    }

    let full = eval_sdf_full(
        py,
        kernels,
        xyz,
        n_pts,
        n_kernels,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
        b_min,
        b_max,
        enclosed,
    )?;

    if n_kernels == 1 {
        return Ok(full);
    }

    let mut min_sdf = vec![f32::INFINITY; n_pts];
    for i in 0..n_pts {
        for k in 0..n_kernels {
            let v = full[i * n_kernels + k];
            if v < min_sdf[i] {
                min_sdf[i] = v;
            }
        }
    }
    Ok(min_sdf)
}

// ---------------------------------------------------------------------------
// Mesh result
// ---------------------------------------------------------------------------

/// Plain-data result for a single element mesh.
pub struct MeshData {
    /// Vertex positions (n_verts × 3, f64).
    pub vertices: Vec<f64>,
    pub n_verts: usize,
    /// Face indices (n_faces × 3, i32).
    pub faces: Vec<i32>,
    pub n_faces: usize,
    /// Per-vertex in-view tag.
    pub in_view_tag: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Per-element mesh construction (matches _construct_element_mesh + _refine_extra_vertices)
// ---------------------------------------------------------------------------

fn construct_element_mesh(
    py: Python<'_>,
    lib: &CoreLib,
    element: i32,
    kernel: &Py<PyAny>,
    num_verts: i32,
    bisection_iters: i32,
    bisection_tol: f64,
    sdf_batch_size: Option<usize>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<MeshData> {
    let nv = num_verts as usize;

    // ---- centers --------------------------------------------------------
    let mut centers = vec![0.0f64; nv * 3];
    py.allow_threads(|| unsafe { (lib.get_verts_center)(element, centers.as_mut_ptr()) });

    let center_sdf = eval_kernel_py(
        py,
        kernel,
        &centers,
        nv,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
    )?;
    if center_sdf.len() != nv {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "center SDF length mismatch: got {}, expected {nv}",
            center_sdf.len()
        )));
    }

    // ---- cube corners (initial, then bisection) --------------------------
    let mut cubes = vec![0.0f64; nv * 8 * 3];
    // First call with null pointers fills cubes with initial corner positions.
    py.allow_threads(|| unsafe {
        (lib.update_verts)(
            element,
            std::ptr::null(),
            std::ptr::null(),
            cubes.as_mut_ptr(),
        )
    });

    let n_cubes = nv * 8;
    let mut sdf_buf = vec![0.0f32; n_cubes];
    let check_tol = bisection_tol > 0.0;

    for _ in 0..bisection_iters {
        let sdf_cubes = eval_kernel_py(
            py,
            kernel,
            &cubes,
            n_cubes,
            sdf_batch_size,
            torch_eval_device,
            torch_eval_dtype,
        )?;
        sdf_buf.copy_from_slice(&sdf_cubes);

        py.allow_threads(|| unsafe {
            (lib.update_verts)(
                element,
                sdf_buf.as_ptr(),
                center_sdf.as_ptr(),
                cubes.as_mut_ptr(),
            )
        });

        if check_tol {
            let max_abs = sdf_buf.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if (max_abs as f64) < bisection_tol {
                break;
            }
        }
    }

    // ---- LR positions + finalize vertex positions -----------------------
    let mut cubes_r = vec![0.0f64; n_cubes * 3];
    py.allow_threads(|| unsafe { (lib.get_lr_verts)(element, cubes.as_mut_ptr(), cubes_r.as_mut_ptr()) });

    let sdf_l = eval_kernel_py(
        py,
        kernel,
        &cubes,
        n_cubes,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
    )?;
    let sdf_r = eval_kernel_py(
        py,
        kernel,
        &cubes_r,
        n_cubes,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
    )?;

    let mut vertices = vec![0.0f64; nv * 3];
    py.allow_threads(|| unsafe {
        (lib.finalize_verts)(
            element,
            sdf_l.as_ptr(),
            sdf_r.as_ptr(),
            vertices.as_mut_ptr(),
        )
    });

    // ---- Extra vertices (edge + face) + faces ---------------------------
    let mut cnts = [0i32; 3];
    py.allow_threads(|| unsafe { (lib.construct_faces)(element, vertices.as_mut_ptr(), cnts.as_mut_ptr()) });
    let (nve, nvf, nf) = (cnts[0] as usize, cnts[1] as usize, cnts[2] as usize);

    let (final_vertices, faces) = if nve == 0 && nvf == 0 {
        let mut f = vec![0i32; nf * 3];
        py.allow_threads(|| unsafe { (lib.get_faces)(f.as_mut_ptr()) });
        (vertices, f)
    } else {
        refine_extra_vertices(
            py, lib, kernel, &vertices, nv, nve, nvf, nf, bisection_iters, bisection_tol,
            sdf_batch_size, torch_eval_device, torch_eval_dtype, b_min, b_max, enclosed,
        )?
    };

    // ---- In-view tag ----------------------------------------------------
    let n_final_verts = final_vertices.len() / 3;
    let mut in_view_tag = vec![false; n_final_verts];
    py.allow_threads(|| unsafe { (lib.get_in_view_tag)(element, in_view_tag.as_mut_ptr()) });

    Ok(MeshData {
        vertices: final_vertices,
        n_verts: n_final_verts,
        faces,
        n_faces: nf,
        in_view_tag,
    })
}

/// Matches `OcMesher._refine_extra_vertices`.
#[allow(clippy::too_many_arguments)]
fn refine_extra_vertices(
    py: Python<'_>,
    lib: &CoreLib,
    kernel: &Py<PyAny>,
    base_vertices: &[f64],
    n_base: usize,
    nve: usize,
    nvf: usize,
    nf: usize,
    bisection_iters: i32,
    bisection_tol: f64,
    sdf_batch_size: Option<usize>,
    torch_eval_device: Option<&str>,
    torch_eval_dtype: Option<&str>,
    _b_min: &[f64; 3],
    _b_max: &[f64; 3],
    _enclosed: bool,
) -> PyResult<(Vec<f64>, Vec<i32>)> {
    let mut edge_centers = vec![0.0f64; nve * 3];
    let mut face_centers = vec![0.0f64; nvf * 3];
    py.allow_threads(|| unsafe { (lib.get_extra_verts_center)(edge_centers.as_mut_ptr(), face_centers.as_mut_ptr()) });

    // Fused center SDF
    let ef_n = nve + nvf;
    let mut ef_centers = vec![0.0f64; ef_n * 3];
    ef_centers[..nve * 3].copy_from_slice(&edge_centers);
    ef_centers[nve * 3..].copy_from_slice(&face_centers);
    let ef_sdf = eval_kernel_py(
        py,
        kernel,
        &ef_centers,
        ef_n,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
    )?;
    let ecenter_sdf = ef_sdf[..nve].to_vec();
    let fcenter_sdf = ef_sdf[nve..].to_vec();

    // Initial LR positions
    let mut edge_lr = vec![0.0f64; nve * 2 * 3];
    let mut face_lr = vec![0.0f64; nvf * 4 * 3];
    py.allow_threads(|| unsafe {
        (lib.update_extra_verts)(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            edge_lr.as_mut_ptr(),
            face_lr.as_mut_ptr(),
        )
    });

    let n_elr = nve * 2;
    let n_flr = nvf * 4;
    let n_bisect = n_elr + n_flr;
    let check_tol = bisection_tol > 0.0;
    let mut bisect_buf = vec![0.0f64; n_bisect * 3];
    let mut sdf_buf = vec![0.0f32; n_bisect];

    for _ in 0..bisection_iters {
        bisect_buf[..n_elr * 3].copy_from_slice(&edge_lr);
        bisect_buf[n_elr * 3..].copy_from_slice(&face_lr);

        let sdf_all = eval_kernel_py(
            py,
            kernel,
            &bisect_buf,
            n_bisect,
            sdf_batch_size,
            torch_eval_device,
            torch_eval_dtype,
        )?;
        sdf_buf.copy_from_slice(&sdf_all);

        let (e_sdf, f_sdf) = sdf_buf.split_at(n_elr);
        py.allow_threads(|| unsafe {
            (lib.update_extra_verts)(
                e_sdf.as_ptr(),
                f_sdf.as_ptr(),
                ecenter_sdf.as_ptr(),
                fcenter_sdf.as_ptr(),
                edge_lr.as_mut_ptr(),
                face_lr.as_mut_ptr(),
            )
        });

        if check_tol {
            let max_abs = sdf_buf.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if (max_abs as f64) < bisection_tol {
                break;
            }
        }
    }

    // Fused LR extra eval
    let mut edge_r = vec![0.0f64; nve * 2 * 3];
    let mut face_r = vec![0.0f64; nvf * 4 * 3];
    py.allow_threads(|| unsafe {
        (lib.get_lr_extra_verts)(
            edge_lr.as_mut_ptr(),
            edge_r.as_mut_ptr(),
            face_lr.as_mut_ptr(),
            face_r.as_mut_ptr(),
        )
    });

    let n_all_lr = n_elr * 2 + n_flr * 2;
    let mut all_lr = vec![0.0f64; n_all_lr * 3];
    all_lr[..n_elr * 3].copy_from_slice(&edge_lr);
    all_lr[n_elr * 3..n_elr * 2 * 3].copy_from_slice(&edge_r);
    all_lr[n_elr * 2 * 3..(n_elr * 2 + n_flr) * 3].copy_from_slice(&face_lr);
    all_lr[(n_elr * 2 + n_flr) * 3..].copy_from_slice(&face_r);

    let all_lr_sdf = eval_kernel_py(
        py,
        kernel,
        &all_lr,
        n_all_lr,
        sdf_batch_size,
        torch_eval_device,
        torch_eval_dtype,
    )?;
    let esdf_l = &all_lr_sdf[..n_elr];
    let esdf_r = &all_lr_sdf[n_elr..n_elr * 2];
    let fsdf_l = &all_lr_sdf[n_elr * 2..n_elr * 2 + n_flr];
    let fsdf_r = &all_lr_sdf[n_elr * 2 + n_flr..];

    let mut edge_verts = vec![0.0f64; nve * 3];
    let mut face_verts = vec![0.0f64; nvf * 3];
    py.allow_threads(|| unsafe {
        (lib.finalize_extra_verts)(
            esdf_l.as_ptr(),
            esdf_r.as_ptr(),
            edge_verts.as_mut_ptr(),
            fsdf_l.as_ptr(),
            fsdf_r.as_ptr(),
            face_verts.as_mut_ptr(),
        )
    });

    // Assemble final vertex array: base | edge | face
    let n_final = n_base + nve + nvf;
    let mut final_v = vec![0.0f64; n_final * 3];
    final_v[..n_base * 3].copy_from_slice(base_vertices);
    final_v[n_base * 3..(n_base + nve) * 3].copy_from_slice(&edge_verts);
    final_v[(n_base + nve) * 3..].copy_from_slice(&face_verts);

    let mut faces = vec![0i32; nf * 3];
    py.allow_threads(|| unsafe { (lib.get_faces)(faces.as_mut_ptr()) });

    Ok((final_v, faces))
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/// Meshing parameters (mirrors `OcMesher.__init__` keyword arguments).
#[derive(Clone, Debug)]
pub struct MesherParams {
    pub cameras_data: Vec<f64>,
    pub n_cams: i32,
    pub center: [f64; 3],
    pub size: f64,
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub pixels_per_cube: f64,
    pub inv_scale: f64,
    pub min_dist: f64,
    pub memory_limit_mb: i32,
    pub bisection_iters: i32,
    pub bisection_tol: f64,
    pub enclosed: bool,
    pub simplify_occluded: bool,
    pub visible_relax_iter: i32,
    pub coarse_count: i32,
    pub sdf_batch_size: Option<usize>,
    pub torch_eval_device: Option<String>,
    pub torch_eval_dtype: Option<String>,
}

/// Run the full OcMesher pipeline.
///
/// Acquires [`CORE_LOCK`] for the duration of the call, then mirrors
/// `OcMesher.__call__` step by step.  Returns one [`MeshData`] per kernel.
pub fn run_meshing_pipeline(
    py: Python<'_>,
    lib: &CoreLib,
    params: &MesherParams,
    kernels: &[Py<PyAny>],
) -> PyResult<Vec<MeshData>> {
    let n_elements = kernels.len() as i32;
    let n_kerns = kernels.len();
    let torch_eval_device = params.torch_eval_device.as_deref();
    let torch_eval_dtype = params.torch_eval_dtype.as_deref();

    let _guard = core_lock()
        .lock()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("core lock poisoned"))?;

    // ------------------------------------------------------------------
    // Step 1: Coarse octree (visibility only, no SDF)
    // ------------------------------------------------------------------
    let mut center_arr = params.center;
    let mut cams_arr = params.cameras_data.clone();

    let _n_blocks = py.allow_threads(|| unsafe {
        (lib.run_coarse)(
            center_arr.as_mut_ptr(),
            params.size,
            params.n_cams,
            cams_arr.as_mut_ptr(),
            params.pixels_per_cube,
            params.inv_scale,
            params.min_dist,
            params.coarse_count,
            params.memory_limit_mb,
            n_elements,
        )
    });

    // ------------------------------------------------------------------
    // Step 2: Coarse SDF evaluation (mark solid/empty)
    // ------------------------------------------------------------------
    loop {
        let inc = py.allow_threads(|| unsafe { (lib.fine_group)() });
        if inc == 0 {
            break;
        }

        // First call with null: fill output_vertices list, return count.
        let mut n = py.allow_threads(|| unsafe { (lib.fine_iteration)(std::ptr::null_mut()) });

        while n > 0 {
            let n_pts = n as usize;
            let mut xyz = vec![0.0f64; n_pts * 3];
            py.allow_threads(|| unsafe { (lib.fine_iteration_output)(xyz.as_mut_ptr()) });

            let mut sdf_min = eval_sdf_min(
                py,
                kernels,
                &xyz,
                n_pts,
                params.sdf_batch_size,
                torch_eval_device,
                torch_eval_dtype,
                &params.bounds_min,
                &params.bounds_max,
                params.enclosed,
            )?;

            n = py.allow_threads(|| unsafe { (lib.fine_iteration)(sdf_min.as_mut_ptr()) });
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Visibility filter
    // ------------------------------------------------------------------
    let _n_vis = py.allow_threads(|| unsafe { (lib.vis_filter)(params.simplify_occluded, params.visible_relax_iter) });

    // ------------------------------------------------------------------
    // Step 4: Fine octree subdivision near the surface
    // ------------------------------------------------------------------
    let mut nv = vec![0i32; n_kerns];

    loop {
        let n = py.allow_threads(|| unsafe { (lib.final_iteration)(nv.as_mut_ptr()) });
        if n == 0 {
            break;
        }
        let n_pts = n as usize;
        let mut xyz = vec![0.0f64; n_pts * 3];
        py.allow_threads(|| unsafe { (lib.final_iteration2)(xyz.as_mut_ptr()) });

        let mut sdf_full = eval_sdf_full(
            py,
            kernels,
            &xyz,
            n_pts,
            n_kerns,
            params.sdf_batch_size,
            torch_eval_device,
            torch_eval_dtype,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
        )?;

        py.allow_threads(|| unsafe { (lib.final_iteration3)(sdf_full.as_mut_ptr()) });
    }

    // Occluded cells
    let n = py.allow_threads(|| unsafe { (lib.final_iteration_occluded)(nv.as_mut_ptr()) });
    if n > 0 {
        let n_pts = n as usize;
        let mut xyz = vec![0.0f64; n_pts * 3];
        py.allow_threads(|| unsafe { (lib.final_iteration2)(xyz.as_mut_ptr()) });

        let mut sdf_full = eval_sdf_full(
            py,
            kernels,
            &xyz,
            n_pts,
            n_kerns,
            params.sdf_batch_size,
            torch_eval_device,
            torch_eval_dtype,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
        )?;

        py.allow_threads(|| unsafe { (lib.final_iteration3_occluded)(sdf_full.as_mut_ptr()) });
    }

    py.allow_threads(|| unsafe { (lib.final_remaining)(nv.as_mut_ptr()) });

    // ------------------------------------------------------------------
    // Step 5: Per-element mesh construction
    // ------------------------------------------------------------------
    let mut results = Vec::with_capacity(n_kerns);
    for (e, kernel) in kernels.iter().enumerate() {
        let mesh = construct_element_mesh(
            py,
            lib,
            e as i32,
            kernel,
            nv[e],
            params.bisection_iters,
            params.bisection_tol,
            params.sdf_batch_size,
            torch_eval_device,
            torch_eval_dtype,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
        )?;
        results.push(mesh);
    }

    Ok(results)
}

/// Convert a [`MeshData`] into a Python `trimesh.Trimesh` object and a numpy bool array.
pub fn mesh_data_to_python<'py>(
    py: Python<'py>,
    mesh: MeshData,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    use ndarray::{Array1, Array2};
    use numpy::IntoPyArray;

    let verts_arr = Array2::from_shape_vec((mesh.n_verts, 3), mesh.vertices)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let faces_arr = Array2::from_shape_vec((mesh.n_faces, 3), mesh.faces)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let tag_arr = Array1::from_vec(mesh.in_view_tag);

    let verts_np = verts_arr.into_pyarray_bound(py);
    let faces_np = faces_arr.into_pyarray_bound(py);
    let tag_np = tag_arr.into_pyarray_bound(py);

    let trimesh_mod = py.import_bound("trimesh")?;
    let trimesh_cls = trimesh_mod.getattr("Trimesh")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("vertices", verts_np)?;
    kwargs.set_item("faces", faces_np)?;
    kwargs.set_item("process", false.into_py(py))?;
    let mesh_obj = trimesh_cls.call(pyo3::types::PyTuple::empty_bound(py), Some(&kwargs))?;

    Ok((mesh_obj, tag_np.into_any()))
}
