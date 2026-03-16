# API Reference

## `OcMesher`

The main C++-backed mesher.  Accepts SDF kernels and camera data, runs
coarse-to-fine octree meshing, and returns trimesh objects.

::: ocmesher.OcMesher

---

## `TorchOcMesher`

Optional PyTorch-backed mesher that keeps tensors on GPU for efficient
batched SDF evaluation.  Requires `torch` to be installed.

::: ocmesher.torch_core.TorchOcMesher
