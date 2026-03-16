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

use std::ffi::c_int;
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

/// Call a single Python SDF kernel with `xyz` (n_pts × 3 f64) and return f32 values.
///
/// Accepts numpy float32, numpy float64, or any Python object that has `.numpy()`.
fn eval_kernel_py_once(
    py: Python<'_>,
    kernel: &Py<PyAny>,
    xyz: &[f64],
    n_pts: usize,
) -> PyResult<Vec<f32>> {
    use numpy::{IntoPyArray, PyReadonlyArray1};
    use ndarray::Array2;

    let xyz_arr = Array2::from_shape_vec((n_pts, 3), xyz.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let xyz_np = xyz_arr.into_pyarray_bound(py);

    let raw = kernel.call1(py, (xyz_np,))?;

    // Try numpy f32 directly
    if let Ok(arr) = raw.extract::<PyReadonlyArray1<f32>>(py) {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
    }

    // Try numpy f64 → cast to f32
    if let Ok(arr) = raw.extract::<PyReadonlyArray1<f64>>(py) {
        return arr
            .as_slice()
            .map(|s| s.iter().map(|&v| v as f32).collect())
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("SDF array not contiguous"));
    }

    // Try `.numpy()` for torch tensors (CPU) or DLPack-capable objects
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

    // Last resort: try __dlpack__ → capsule → numpy (zero-copy path for CPU DLPack)
    if let Ok(cap) = raw.bind(py).call_method0("__dlpack__") {
        // Import numpy and call np.from_dlpack
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

fn eval_kernel_py(
    py: Python<'_>,
    kernel: &Py<PyAny>,
    xyz: &[f64],
    n_pts: usize,
    sdf_batch_size: Option<usize>,
) -> PyResult<Vec<f32>> {
    let Some(batch_size) = sdf_batch_size else {
        return eval_kernel_py_once(py, kernel, xyz, n_pts);
    };

    if batch_size == 0 || n_pts <= batch_size {
        return eval_kernel_py_once(py, kernel, xyz, n_pts);
    }

    let mut out = Vec::with_capacity(n_pts);
    for start in (0..n_pts).step_by(batch_size) {
        let end = (start + batch_size).min(n_pts);
        let chunk = &xyz[start * 3..end * 3];
        let chunk_vals = eval_kernel_py_once(py, kernel, chunk, end - start)?;
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
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<Vec<f32>> {
    let mut result = vec![0.0f32; n_pts * n_kernels];

    for (k_idx, kernel) in kernels.iter().enumerate() {
        let sdf = eval_kernel_py(py, kernel, xyz, n_pts, sdf_batch_size)?;
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
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<Vec<f32>> {
    let n_kernels = kernels.len();
    let full = eval_sdf_full(
        py,
        kernels,
        xyz,
        n_pts,
        n_kernels,
        sdf_batch_size,
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
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> PyResult<MeshData> {
    let nv = num_verts as usize;

    // ---- centers --------------------------------------------------------
    let mut centers = vec![0.0f64; nv * 3];
    py.allow_threads(|| unsafe { (lib.get_verts_center)(element, centers.as_mut_ptr()) });

    let center_sdf = eval_kernel_py(py, kernel, &centers, nv, sdf_batch_size)?;
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
        let sdf_cubes = eval_kernel_py(py, kernel, &cubes, n_cubes, sdf_batch_size)?;
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

    let sdf_l = eval_kernel_py(py, kernel, &cubes, n_cubes, sdf_batch_size)?;
    let sdf_r = eval_kernel_py(py, kernel, &cubes_r, n_cubes, sdf_batch_size)?;

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
            sdf_batch_size, b_min, b_max, enclosed,
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
    let ef_sdf = eval_kernel_py(py, kernel, &ef_centers, ef_n, sdf_batch_size)?;
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

        let sdf_all = eval_kernel_py(py, kernel, &bisect_buf, n_bisect, sdf_batch_size)?;
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

    let all_lr_sdf = eval_kernel_py(py, kernel, &all_lr, n_all_lr, sdf_batch_size)?;
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
