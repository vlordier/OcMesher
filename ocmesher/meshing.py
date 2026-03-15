# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Mesh construction helpers: vertex bisection and face assembly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
from tqdm import tqdm

from .utils.interface import AC, POINTER, as_bool, as_int

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .dll import CoreDLL

logger = logging.getLogger(__name__)


def bisect_cube_vertices(
    dll: CoreDLL,
    element_idx: int,
    kernel: list[Callable],
    nv_count: int,
    *,
    eval_fn: Callable[..., NDArray],
    as_flt: Callable,
    as_sdf_flt: Callable,
    sdf_ctype: Any,
    np_float: type,
    bisection_iters: int,
) -> NDArray:
    """Refine cube vertex positions via iterative bisection.

    Args:
        dll: Loaded C++ library wrapper.
        element_idx: Index of the current SDF element.
        kernel: Single-element kernel list for SDF evaluation.
        nv_count: Number of cube vertices.
        eval_fn: SDF evaluation function (kernels, positions) -> (N, K).
        as_flt: Converter for float arrays to ctypes pointers.
        as_sdf_flt: Converter for SDF float arrays to ctypes pointers.
        sdf_ctype: ctypes SDF float type (e.g. ``c_float``).
        np_float: NumPy float dtype (e.g. ``np.float64``).
        bisection_iters: Number of bisection iterations.

    Returns:
        (nv_count, 3) refined vertex positions.
    """
    # np.empty avoids zero-init since the C function fills these immediately.
    centers = np.empty((nv_count, 3), dtype=np_float)
    dll.get_verts_center(element_idx, as_flt(centers))
    center_sdf = eval_fn(kernel, centers)

    cubes = AC(np.empty((nv_count * 8, 3), dtype=np_float))
    null_sdf = POINTER(sdf_ctype)()
    dll.update_verts(element_idx, null_sdf, null_sdf, as_flt(cubes))
    center_sdf_ptr = as_sdf_flt(AC(center_sdf))

    for _ in tqdm(range(bisection_iters)):
        sdf = eval_fn(kernel, cubes)
        dll.update_verts(
            element_idx,
            as_sdf_flt(AC(sdf)),
            center_sdf_ptr,
            as_flt(cubes),
        )

    cubes_r = AC(np.empty((nv_count * 8, 3), dtype=np_float))
    dll.get_lr_verts(element_idx, as_flt(cubes), as_flt(cubes_r))

    # Fused left/right SDF evaluation: single eval_fn call instead
    # of two, halving the Python→SDF round-trip overhead.
    n_cubes = len(cubes)
    lr_combined = np.concatenate([cubes, cubes_r])
    lr_sdf = eval_fn(kernel, lr_combined)
    sdf_l = lr_sdf[:n_cubes]
    sdf_r = lr_sdf[n_cubes:]
    del cubes, cubes_r, centers, center_sdf, lr_combined, lr_sdf

    # np.empty is safe: finalize_verts writes every element before use.
    vertices = np.empty((nv_count, 3), dtype=np_float)
    dll.finalize_verts(
        element_idx,
        as_sdf_flt(sdf_l),
        as_sdf_flt(sdf_r),
        as_flt(vertices),
    )
    return vertices


def bisect_extra_vertices(
    dll: CoreDLL,
    kernel: list[Callable],
    nve: int,
    nvf: int,
    *,
    eval_fn: Callable[..., NDArray],
    as_flt: Callable,
    as_sdf_flt: Callable,
    sdf_ctype: Any,
    np_float: type,
    bisection_iters: int,
) -> tuple[NDArray, NDArray]:
    """Refine edge and face vertex positions via iterative bisection.

    Optimisations:
    - Fused edge + face SDF evaluation per bisection iteration: a single
      ``eval_fn`` call replaces two, halving Python→SDF round trips
      in the inner loop (30 calls → 15 for the default 15 iterations).
    - Fused left/right SDF evaluation after bisection: 4 calls → 2.

    Args:
        dll: Loaded C++ library wrapper.
        kernel: Single-element kernel list for SDF evaluation.
        nve: Number of edge vertices.
        nvf: Number of face vertices.
        eval_fn: SDF evaluation function.
        as_flt: Float array converter.
        as_sdf_flt: SDF float array converter.
        sdf_ctype: ctypes SDF float type.
        np_float: NumPy float dtype.
        bisection_iters: Number of bisection iterations.

    Returns:
        Tuple of (edge_vertices, face_vertices), each (N, 3).
    """
    # np.empty avoids zero-init: C functions fill all elements immediately.
    edge_c = AC(np.empty((nve, 3), dtype=np_float))
    face_c = AC(np.empty((nvf, 3), dtype=np_float))
    dll.get_extra_verts_center(as_flt(edge_c), as_flt(face_c))

    # Fused center SDF: one eval_fn call for edge + face centres.
    n_edge_c = len(edge_c)
    ef_centers = np.concatenate([edge_c, face_c])
    ef_center_sdf = eval_fn(kernel, ef_centers)
    ecenter_sdf = ef_center_sdf[:n_edge_c]
    fcenter_sdf = ef_center_sdf[n_edge_c:]
    del ef_centers, ef_center_sdf

    edge_lr = AC(np.empty((nve * 2, 3), dtype=np_float))
    face_lr = AC(np.empty((nvf * 4, 3), dtype=np_float))
    null_sdf = POINTER(sdf_ctype)()
    dll.update_extra_verts(
        null_sdf,
        null_sdf,
        null_sdf,
        null_sdf,
        as_flt(edge_lr),
        as_flt(face_lr),
    )

    n_edge_lr = len(edge_lr)
    for _ in range(bisection_iters):
        # Fused edge + face SDF: one call instead of two per iteration.
        ef_combined = np.concatenate([edge_lr, face_lr])
        ef_sdf = eval_fn(kernel, ef_combined)
        e_sdf = ef_sdf[:n_edge_lr]
        f_sdf = ef_sdf[n_edge_lr:]
        dll.update_extra_verts(
            as_sdf_flt(e_sdf),
            as_sdf_flt(f_sdf),
            as_sdf_flt(ecenter_sdf),
            as_sdf_flt(fcenter_sdf),
            as_flt(edge_lr),
            as_flt(face_lr),
        )

    del edge_c, face_c, ecenter_sdf, fcenter_sdf
    edge_r = AC(np.empty((nve * 2, 3), dtype=np_float))
    face_r = AC(np.empty((nvf * 4, 3), dtype=np_float))
    dll.get_lr_extra_verts(
        as_flt(edge_lr),
        as_flt(edge_r),
        as_flt(face_lr),
        as_flt(face_r),
    )

    # Fused left/right SDF: 2 calls instead of 4.
    n_elr = len(edge_lr)
    e_combined = np.concatenate([edge_lr, edge_r])
    e_sdf_all = eval_fn(kernel, e_combined)
    esdf_l = e_sdf_all[:n_elr]
    esdf_r = e_sdf_all[n_elr:]

    n_flr = len(face_lr)
    f_combined = np.concatenate([face_lr, face_r])
    f_sdf_all = eval_fn(kernel, f_combined)
    fsdf_l = f_sdf_all[:n_flr]
    fsdf_r = f_sdf_all[n_flr:]

    del edge_lr, edge_r, face_lr, face_r

    edge_verts = np.empty((nve, 3), dtype=np_float)
    face_verts = np.empty((nvf, 3), dtype=np_float)
    dll.finalize_extra_verts(
        as_sdf_flt(esdf_l),
        as_sdf_flt(esdf_r),
        as_flt(edge_verts),
        as_sdf_flt(fsdf_l),
        as_sdf_flt(fsdf_r),
        as_flt(face_verts),
    )
    return edge_verts, face_verts


def construct_element_mesh(
    dll: CoreDLL,
    element_idx: int,
    kernel: list[Callable],
    nv_count: int,
    *,
    eval_fn: Callable[..., NDArray],
    as_flt: Callable,
    as_sdf_flt: Callable,
    sdf_ctype: Any,
    np_float: type,
    bisection_iters: int,
) -> tuple[trimesh.Trimesh, NDArray]:
    """Build a single element's triangle mesh.

    Combines vertex bisection, face construction, and extra-vertex refinement
    into the final mesh.

    Returns:
        Tuple of (mesh, in_view_tag).
    """
    bisect_kw = {
        "eval_fn": eval_fn,
        "as_flt": as_flt,
        "as_sdf_flt": as_sdf_flt,
        "sdf_ctype": sdf_ctype,
        "np_float": np_float,
        "bisection_iters": bisection_iters,
    }

    vertices = bisect_cube_vertices(
        dll,
        element_idx,
        kernel,
        nv_count,
        **bisect_kw,
    )

    cnts = np.zeros(3, dtype=np.int32)
    dll.construct_faces(element_idx, as_flt(vertices), as_int(cnts))
    nve, nvf, nf = int(cnts[0]), int(cnts[1]), int(cnts[2])

    edge_verts, face_verts = bisect_extra_vertices(
        dll,
        kernel,
        nve,
        nvf,
        **bisect_kw,
    )

    faces = AC(np.empty((nf, 3), dtype=np.int32))
    dll.get_faces(as_int(faces))
    vertices = np.concatenate((vertices, edge_verts, face_verts))

    in_view_tag = np.zeros(vertices.shape[0], dtype=bool)
    dll.get_in_view_tag(element_idx, as_bool(in_view_tag))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    logger.info(
        "element %d: %d vertices, %d faces",
        element_idx,
        mesh.vertices.shape[0],
        mesh.faces.shape[0],
    )
    return mesh, in_view_tag
