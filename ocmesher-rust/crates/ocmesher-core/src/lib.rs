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
use std::time::Instant;

use libloading::{Library, Symbol};
// Removed all PyO3 dependencies for Rust-native pipeline

mod native_kernels;

#[cfg(feature = "tch-kernels")]
pub use native_kernels::tch_kernels;
pub use native_kernels::{build_native_kernels, PlaneSpec, PrimitiveSpec, SphereSpec};
pub use native_kernels::{BoxedSdfEvaluator, PlaneKernel, SdfEvaluator, SphereKernel};

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

// Removed PyErr conversion, not needed for Rust-native pipeline

// ---------------------------------------------------------------------------
// Process-wide serialisation lock (C++ global state)
// ---------------------------------------------------------------------------

static CORE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn core_lock() -> &'static Mutex<()> {
    CORE_LOCK.get_or_init(|| Mutex::new(()))
}

fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OCMESHER_PROFILE")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

macro_rules! profile_start {
    () => {
        Instant::now()
    };
}

macro_rules! profile_print {
    ($start:expr, $label:expr) => {
        if profile_enabled() {
            eprintln!(
                "[profile] {:>30}: {:>8.2}ms",
                $label,
                $start.elapsed().as_secs_f64() * 1000.0
            );
        }
    };
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
type FnFinalIteration = unsafe extern "C" fn() -> c_int;
type FnFinalIterationOccluded = unsafe extern "C" fn() -> c_int;
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
                fine_iteration_output: load_sym!(
                    lib,
                    "fine_iteration_output",
                    FnFineIterationOutput
                ),
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

/// Validate and extract scalar bounds from a 6-element array.
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

// SDF evaluation helpers for Rust-native pipeline are in native_kernels.rs

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
    pub fine_batch_size: i32,
}

/// Run the full OcMesher pipeline using Rust-native SDFs only.
pub fn run_meshing_pipeline_native(
    lib: &CoreLib,
    params: &MesherParams,
    kernels: &[BoxedSdfEvaluator],
) -> Result<Vec<MeshData>, CoreError> {
    let t_total = profile_start!();
    let n_elements = kernels.len() as i32;
    let n_kerns = kernels.len();

    let _guard = core_lock()
        .lock()
        .map_err(|_| CoreError::Sdf("core lock poisoned".to_string()))?;

    let mut center_arr = params.center;
    let mut cams_arr = params.cameras_data.clone();

    let t = profile_start!();
    let _n_blocks = unsafe {
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
    };
    profile_print!(t, "run_coarse");

    let mut fine_xyz = Vec::<f64>::new();
    let mut fine_sdf = Vec::<f32>::new();

    let t = profile_start!();
    let mut fine_iters = 0u32;
    let mut fine_pts_total = 0usize;
    let mut fine_cpp_us = 0u64;
    let mut fine_sdf_us = 0u64;
    let mut fine_group_us = 0u64;
    let mut fine_iter_us = 0u64;
    let mut fine_output_us = 0u64;
    let fine_batch_size = params.fine_batch_size.max(1) as usize;
    loop {
        let tc = profile_start!();
        let inc = unsafe { (lib.fine_group)() };
        let dt_group = tc.elapsed().as_micros() as u64;
        fine_group_us += dt_group;
        if inc == 0 {
            fine_cpp_us += dt_group;
            break;
        }

        let tc_iter0 = profile_start!();
        let mut n = unsafe { (lib.fine_iteration)(std::ptr::null_mut()) };
        let dt_iter0 = tc_iter0.elapsed().as_micros() as u64;
        fine_iter_us += dt_iter0;
        fine_cpp_us += dt_iter0;
        while n > 0 {
            let n_pts = n as usize;
            fine_iters += 1;
            fine_pts_total += n_pts;
            fine_xyz.resize(n_pts * 3, 0.0);
            let tc_out = profile_start!();
            unsafe { (lib.fine_iteration_output)(fine_xyz.as_mut_ptr()) };
            let dt_out = tc_out.elapsed().as_micros() as u64;
            fine_output_us += dt_out;
            fine_cpp_us += dt_out;

            let ts = profile_start!();
            native_kernels::eval_sdf_min_native_into(
                kernels,
                &fine_xyz,
                n_pts,
                &params.bounds_min,
                &params.bounds_max,
                params.enclosed,
                &mut fine_sdf,
            )?;
            fine_sdf_us += ts.elapsed().as_micros() as u64;

            let tc2 = profile_start!();
            n = unsafe { (lib.fine_iteration)(fine_sdf.as_mut_ptr()) };
            let dt_iter = tc2.elapsed().as_micros() as u64;
            fine_iter_us += dt_iter;
            fine_cpp_us += dt_iter;
        }
    }
    if profile_enabled() {
        eprintln!("[profile] {:>30}: {:>8.2}ms  ({} iters, {} pts, cpp={:.2}ms, sdf={:.2}ms, fine_group={:.2}ms, fine_iter={:.2}ms, fine_output={:.2}ms, batch_size={})",
            "fine_loop", t.elapsed().as_secs_f64() * 1000.0, fine_iters, fine_pts_total,
            fine_cpp_us as f64 / 1000.0, fine_sdf_us as f64 / 1000.0,
            fine_group_us as f64 / 1000.0, fine_iter_us as f64 / 1000.0, fine_output_us as f64 / 1000.0,
            fine_batch_size);
    }

    let t = profile_start!();
    let _n_vis = unsafe { (lib.vis_filter)(params.simplify_occluded, params.visible_relax_iter) };
    profile_print!(t, "vis_filter");

    let mut nv = vec![0i32; n_kerns];
    let mut final_xyz = Vec::<f64>::new();
    let mut final_sdf = Vec::<f32>::new();
    let t = profile_start!();
    let mut final_iters = 0u32;
    let mut final_pts_total = 0usize;
    let mut final_cpp_us = 0u64;
    let mut final_sdf_us = 0u64;
    let mut final_iter_us = 0u64;
    let mut final_output_us = 0u64;
    let mut final_push_us = 0u64;
    let mut final_occluded_check_us = 0u64;
    loop {
        let tc = profile_start!();
        let n = unsafe { (lib.final_iteration)() };
        let dt_iter = tc.elapsed().as_micros() as u64;
        final_iter_us += dt_iter;
        if n == 0 {
            final_cpp_us += dt_iter;
            break;
        }
        let n_pts = n as usize;
        final_iters += 1;
        final_pts_total += n_pts;
        final_xyz.resize(n_pts * 3, 0.0);
        let tc_out = profile_start!();
        unsafe { (lib.final_iteration2)(final_xyz.as_mut_ptr()) };
        let dt_out = tc_out.elapsed().as_micros() as u64;
        final_output_us += dt_out;
        final_cpp_us += dt_iter + dt_out;

        let ts = profile_start!();
        native_kernels::eval_sdf_full_native_into(
            kernels,
            &final_xyz,
            n_pts,
            n_kerns,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
            &mut final_sdf,
        )?;
        final_sdf_us += ts.elapsed().as_micros() as u64;

        let tc2 = profile_start!();
        unsafe { (lib.final_iteration3)(final_sdf.as_mut_ptr()) };
        let dt_push = tc2.elapsed().as_micros() as u64;
        final_push_us += dt_push;
        final_cpp_us += dt_push;
    }

    let tc = profile_start!();
    let n = unsafe { (lib.final_iteration_occluded)() };
    let dt_occ = tc.elapsed().as_micros() as u64;
    final_occluded_check_us += dt_occ;
    final_cpp_us += dt_occ;
    if n > 0 {
        let n_pts = n as usize;
        final_iters += 1;
        final_pts_total += n_pts;
        final_xyz.resize(n_pts * 3, 0.0);
        let tc_out = profile_start!();
        unsafe { (lib.final_iteration2)(final_xyz.as_mut_ptr()) };
        let dt_out = tc_out.elapsed().as_micros() as u64;
        final_output_us += dt_out;
        final_cpp_us += dt_out;

        let ts = profile_start!();
        native_kernels::eval_sdf_full_native_into(
            kernels,
            &final_xyz,
            n_pts,
            n_kerns,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
            &mut final_sdf,
        )?;
        final_sdf_us += ts.elapsed().as_micros() as u64;

        let tc2 = profile_start!();
        unsafe { (lib.final_iteration3_occluded)(final_sdf.as_mut_ptr()) };
        let dt_push = tc2.elapsed().as_micros() as u64;
        final_push_us += dt_push;
        final_cpp_us += dt_push;
    }
    if profile_enabled() {
        eprintln!("[profile] {:>30}: {:>8.2}ms  ({} iters, {} pts, cpp={:.2}ms, sdf={:.2}ms, final_iter={:.2}ms, final_output={:.2}ms, final_push={:.2}ms, final_occluded={:.2}ms)",
            "final_loop", t.elapsed().as_secs_f64() * 1000.0, final_iters, final_pts_total,
            final_cpp_us as f64 / 1000.0, final_sdf_us as f64 / 1000.0,
            final_iter_us as f64 / 1000.0, final_output_us as f64 / 1000.0,
            final_push_us as f64 / 1000.0, final_occluded_check_us as f64 / 1000.0);
    }

    unsafe { (lib.final_remaining)(nv.as_mut_ptr()) };

    let t = profile_start!();
    let mut results = Vec::with_capacity(n_kerns);
    for (element, kernel) in kernels.iter().enumerate() {
        let mesh = construct_element_mesh_native(
            lib,
            element as i32,
            kernel.as_ref(),
            nv[element],
            params.bisection_iters,
            params.bisection_tol,
            &params.bounds_min,
            &params.bounds_max,
            params.enclosed,
        )?;
        results.push(mesh);
    }
    if profile_enabled() {
        eprintln!(
            "[profile] {:>30}: {:>8.2}ms  ({} elements, nv={:?})",
            "construct_meshes",
            t.elapsed().as_secs_f64() * 1000.0,
            n_kerns,
            &nv
        );
    }
    profile_print!(t_total, "TOTAL pipeline");

    Ok(results)
}

fn construct_element_mesh_native(
    lib: &CoreLib,
    element: i32,
    kernel: &dyn SdfEvaluator,
    num_verts: i32,
    bisection_iters: i32,
    bisection_tol: f64,
    b_min: &[f64; 3],
    b_max: &[f64; 3],
    enclosed: bool,
) -> Result<MeshData, CoreError> {
    let t_elem = profile_start!();
    let nv = num_verts as usize;

    let mut centers = vec![0.0f64; nv * 3];
    unsafe { (lib.get_verts_center)(element, centers.as_mut_ptr()) };

    let mut center_sdf = Vec::with_capacity(nv);
    kernel.evaluate_batch_into(&centers, nv, &mut center_sdf)?;
    if center_sdf.len() != nv {
        return Err(CoreError::Sdf(format!(
            "native center SDF length mismatch: got {}, expected {nv}",
            center_sdf.len()
        )));
    }

    let mut cubes = vec![0.0f64; nv * 8 * 3];
    unsafe {
        (lib.update_verts)(
            element,
            std::ptr::null(),
            std::ptr::null(),
            cubes.as_mut_ptr(),
        )
    };

    let n_cubes = nv * 8;
    let mut sdf_buf = Vec::with_capacity(n_cubes);
    let check_tol = bisection_tol > 0.0;

    let t_bisect = profile_start!();
    let mut t_sdf_us = 0u64;
    let mut t_cpp_us = 0u64;
    for _ in 0..bisection_iters {
        let ts = profile_start!();
        kernel.evaluate_batch_into(&cubes, n_cubes, &mut sdf_buf)?;
        t_sdf_us += ts.elapsed().as_micros() as u64;
        if sdf_buf.len() != n_cubes {
            return Err(CoreError::Sdf(format!(
                "native cube SDF length mismatch: got {}, expected {n_cubes}",
                sdf_buf.len()
            )));
        }

        let tc = profile_start!();
        unsafe {
            (lib.update_verts)(
                element,
                sdf_buf.as_ptr(),
                center_sdf.as_ptr(),
                cubes.as_mut_ptr(),
            )
        };
        t_cpp_us += tc.elapsed().as_micros() as u64;

        if check_tol {
            let max_abs = sdf_buf
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs()));
            if (max_abs as f64) < bisection_tol {
                break;
            }
        }
    }
    if profile_enabled() {
        eprintln!(
            "[profile] {:>30}: {:>8.2}ms  (sdf={:.2}ms, cpp={:.2}ms, n_cubes={})",
            "bisection_loop",
            t_bisect.elapsed().as_secs_f64() * 1000.0,
            t_sdf_us as f64 / 1000.0,
            t_cpp_us as f64 / 1000.0,
            n_cubes
        );
    }

    let t_lr = profile_start!();
    let mut cubes_r = vec![0.0f64; n_cubes * 3];
    unsafe { (lib.get_lr_verts)(element, cubes.as_mut_ptr(), cubes_r.as_mut_ptr()) };

    // Combine L and R coordinates into one contiguous buffer for a single SDF batch eval.
    let n_lr = n_cubes * 2;
    let mut lr_xyz = vec![0.0f64; n_lr * 3];
    lr_xyz[..n_cubes * 3].copy_from_slice(&cubes[..n_cubes * 3]);
    lr_xyz[n_cubes * 3..].copy_from_slice(&cubes_r[..n_cubes * 3]);

    let mut lr_sdf = Vec::with_capacity(n_lr);
    kernel.evaluate_batch_into(&lr_xyz, n_lr, &mut lr_sdf)?;
    if lr_sdf.len() != n_lr {
        return Err(CoreError::Sdf(format!(
            "native LR SDF length mismatch: got {}, expected {n_lr}",
            lr_sdf.len()
        )));
    }

    let (sdf_l, sdf_r) = lr_sdf.split_at(n_cubes);

    let mut vertices = vec![0.0f64; nv * 3];
    unsafe {
        (lib.finalize_verts)(
            element,
            sdf_l.as_ptr(),
            sdf_r.as_ptr(),
            vertices.as_mut_ptr(),
        )
    };
    profile_print!(t_lr, "lr_finalize_verts");

    let t_faces = profile_start!();

    let mut cnts = [0i32; 3];
    unsafe { (lib.construct_faces)(element, vertices.as_mut_ptr(), cnts.as_mut_ptr()) };
    let (nve, nvf, nf) = (cnts[0] as usize, cnts[1] as usize, cnts[2] as usize);

    let (final_vertices, faces) = if nve == 0 && nvf == 0 {
        let mut raw_faces = vec![0i32; nf * 3];
        unsafe { (lib.get_faces)(raw_faces.as_mut_ptr()) };
        (vertices, raw_faces)
    } else {
        refine_extra_vertices_native(
            lib,
            kernel,
            &vertices,
            nv,
            nve,
            nvf,
            nf,
            bisection_iters,
            bisection_tol,
            b_min,
            b_max,
            enclosed,
        )?
    };

    let n_final_verts = final_vertices.len() / 3;
    let mut in_view_tag = vec![false; n_final_verts];
    unsafe { (lib.get_in_view_tag)(element, in_view_tag.as_mut_ptr()) };

    if profile_enabled() {
        eprintln!(
            "[profile] {:>30}: {:>8.2}ms",
            "construct_faces+extra",
            t_faces.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!(
            "[profile] {:>30}: {:>8.2}ms  (nv={}, nf={}, nve={}, nvf={})",
            "element_total",
            t_elem.elapsed().as_secs_f64() * 1000.0,
            nv,
            nf,
            nve,
            nvf
        );
    }

    Ok(MeshData {
        vertices: final_vertices,
        n_verts: n_final_verts,
        faces,
        n_faces: nf,
        in_view_tag,
    })
}

#[allow(clippy::too_many_arguments)]
fn refine_extra_vertices_native(
    lib: &CoreLib,
    kernel: &dyn SdfEvaluator,
    base_vertices: &[f64],
    n_base: usize,
    nve: usize,
    nvf: usize,
    nf: usize,
    bisection_iters: i32,
    bisection_tol: f64,
    _b_min: &[f64; 3],
    _b_max: &[f64; 3],
    _enclosed: bool,
) -> Result<(Vec<f64>, Vec<i32>), CoreError> {
    let mut edge_centers = vec![0.0f64; nve * 3];
    let mut face_centers = vec![0.0f64; nvf * 3];
    unsafe { (lib.get_extra_verts_center)(edge_centers.as_mut_ptr(), face_centers.as_mut_ptr()) };

    let ef_n = nve + nvf;
    let mut ef_centers = vec![0.0f64; ef_n * 3];
    ef_centers[..nve * 3].copy_from_slice(&edge_centers);
    ef_centers[nve * 3..].copy_from_slice(&face_centers);
    let mut ef_sdf = Vec::with_capacity(ef_n);
    kernel.evaluate_batch_into(&ef_centers, ef_n, &mut ef_sdf)?;
    if ef_sdf.len() != ef_n {
        return Err(CoreError::Sdf(format!(
            "native extra center SDF length mismatch: got {}, expected {ef_n}",
            ef_sdf.len()
        )));
    }
    let mut ecenter_sdf = vec![0.0f32; nve];
    let mut fcenter_sdf = vec![0.0f32; nvf];
    ecenter_sdf.copy_from_slice(&ef_sdf[..nve]);
    fcenter_sdf.copy_from_slice(&ef_sdf[nve..]);

    let mut edge_lr = vec![0.0f64; nve * 2 * 3];
    let mut face_lr = vec![0.0f64; nvf * 4 * 3];
    unsafe {
        (lib.update_extra_verts)(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            edge_lr.as_mut_ptr(),
            face_lr.as_mut_ptr(),
        )
    };

    let n_elr = nve * 2;
    let n_flr = nvf * 4;
    let n_bisect = n_elr + n_flr;
    let check_tol = bisection_tol > 0.0;
    let mut bisect_buf = vec![0.0f64; n_bisect * 3];
    let mut sdf_buf = Vec::with_capacity(n_bisect);

    for _ in 0..bisection_iters {
        bisect_buf[..n_elr * 3].copy_from_slice(&edge_lr);
        bisect_buf[n_elr * 3..].copy_from_slice(&face_lr);

        kernel.evaluate_batch_into(&bisect_buf, n_bisect, &mut sdf_buf)?;
        if sdf_buf.len() != n_bisect {
            return Err(CoreError::Sdf(format!(
                "native extra bisection SDF length mismatch: got {}, expected {n_bisect}",
                sdf_buf.len()
            )));
        }

        let (e_sdf, f_sdf) = sdf_buf.split_at(n_elr);
        unsafe {
            (lib.update_extra_verts)(
                e_sdf.as_ptr(),
                f_sdf.as_ptr(),
                ecenter_sdf.as_ptr(),
                fcenter_sdf.as_ptr(),
                edge_lr.as_mut_ptr(),
                face_lr.as_mut_ptr(),
            )
        };

        if check_tol {
            let max_abs = sdf_buf
                .iter()
                .fold(0.0f32, |acc, &value| acc.max(value.abs()));
            if (max_abs as f64) < bisection_tol {
                break;
            }
        }
    }

    let mut edge_r = vec![0.0f64; nve * 2 * 3];
    let mut face_r = vec![0.0f64; nvf * 4 * 3];
    unsafe {
        (lib.get_lr_extra_verts)(
            edge_lr.as_mut_ptr(),
            edge_r.as_mut_ptr(),
            face_lr.as_mut_ptr(),
            face_r.as_mut_ptr(),
        )
    };

    let n_all_lr = n_elr * 2 + n_flr * 2;
    let mut all_lr = vec![0.0f64; n_all_lr * 3];
    all_lr[..n_elr * 3].copy_from_slice(&edge_lr);
    all_lr[n_elr * 3..n_elr * 2 * 3].copy_from_slice(&edge_r);
    all_lr[n_elr * 2 * 3..(n_elr * 2 + n_flr) * 3].copy_from_slice(&face_lr);
    all_lr[(n_elr * 2 + n_flr) * 3..].copy_from_slice(&face_r);

    let mut all_lr_sdf = Vec::with_capacity(n_all_lr);
    kernel.evaluate_batch_into(&all_lr, n_all_lr, &mut all_lr_sdf)?;
    if all_lr_sdf.len() != n_all_lr {
        return Err(CoreError::Sdf(format!(
            "native LR-extra SDF length mismatch: got {}, expected {n_all_lr}",
            all_lr_sdf.len()
        )));
    }
    let esdf_l = &all_lr_sdf[..n_elr];
    let esdf_r = &all_lr_sdf[n_elr..n_elr * 2];
    let fsdf_l = &all_lr_sdf[n_elr * 2..n_elr * 2 + n_flr];
    let fsdf_r = &all_lr_sdf[n_elr * 2 + n_flr..];

    let mut edge_verts = vec![0.0f64; nve * 3];
    let mut face_verts = vec![0.0f64; nvf * 3];
    unsafe {
        (lib.finalize_extra_verts)(
            esdf_l.as_ptr(),
            esdf_r.as_ptr(),
            edge_verts.as_mut_ptr(),
            fsdf_l.as_ptr(),
            fsdf_r.as_ptr(),
            face_verts.as_mut_ptr(),
        )
    };

    let n_final = n_base + nve + nvf;
    let mut final_vertices = vec![0.0f64; n_final * 3];
    final_vertices[..n_base * 3].copy_from_slice(base_vertices);
    final_vertices[n_base * 3..(n_base + nve) * 3].copy_from_slice(&edge_verts);
    final_vertices[(n_base + nve) * 3..].copy_from_slice(&face_verts);

    let mut faces = vec![0i32; nf * 3];
    unsafe { (lib.get_faces)(faces.as_mut_ptr()) };

    Ok((final_vertices, faces))
}

// mesh_data_to_python removed: all output is now Rust-native
