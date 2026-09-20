# CUDA ray tracing through OptiX

The headless `cuda` backend optionally uses NVIDIA OptiX for triangle GAS/IAS
construction and RT-core traversal. It shares Sparkium's HLSL path tracer,
material graphs, lighting, MIS, transparent shadows, film and sampling with the
CPU/CUDA software pipeline. It does not delegate rendering to Vulkan or run the software
BVH when the OptiX pipeline is selected.

## Build

Requirements: an NVIDIA GPU with RT cores and a supported driver, CUDA with NVRTC, Slang, and the
[official NVIDIA OptiX SDK headers](https://github.com/NVIDIA/optix-dev) version
8.0 or newer. OptiX 9.0 and an RTX 3090 Ti are used for validation. The driver
provides the OptiX runtime; no separate OptiX binary library is linked.

```sh
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release -DVCPKG_PATH=/path/to/vcpkg \
  -DLONGMARCH_ENABLE_OPTIX=ON -DOptiX_ROOT=/path/to/optix-sdk
cmake --build build-release --target demo_sparkium_cli sparkium_native_test -j8
```

`LONGMARCH_OPTIX_INCLUDE_DIR` may be set directly instead of `OptiX_ROOT`.
OptiX is enabled by default when native rendering is enabled and Slang, CUDA,
and the SDK headers are found. Explicitly
enabling it without Slang, CUDA or SDK headers fails configuration with an
actionable error. Use `-DLONGMARCH_ENABLE_OPTIX=OFF` to exclude CUDA hardware ray tracing;
CPU, CUDA and existing graphics backends remain available as before.

NVRTC must generate PTX supported by the installed driver. A newer toolkit's
NVRTC can produce `unsupported toolchain` errors even if CUDA device discovery
succeeds. Use a driver-compatible NVRTC installation (and matching builtins)
or update the driver. A per-process `LD_LIBRARY_PATH` can select that library
on Linux without changing system packages.
Alternatively, configure `CUDA_nvrtc_LIBRARY` and `CUDA_nvrtc_builtins_LIBRARY`
with the full paths to compatible shared libraries, so the build's runtime
search path selects them without a per-command environment override.

## Run

```sh
build-release/demo/sparkium_cli/demo_sparkium_cli \
  assets/scenes/texture/scene.json --backend cuda --pipeline ray_tracing --require-hardware-rt \
  --frames 32 -o optix.png --linear-output optix.pfm \
  --profile optix.csv --profile-cpu-only
```

For this scene, 32 frames × 16 samples/dispatch = 512 spp.
OptiX is the CUDA implementation of `ray_tracing`, not a separate backend or
public pipeline identifier:

- `--backend cuda --pipeline ray_tracing`: OptiX hardware tracing. Fails if the
  SDK was not compiled in or the selected device/driver cannot initialize OptiX.
- `--backend cuda --pipeline rt_fallback`: CUDA kernels with software BVH traversal.
- `--backend cuda --pipeline auto`: selects `ray_tracing` when available, otherwise
  `rt_fallback`. Use this when hardware traversal is optional.

Legacy JSON `ray_tracing` requests use the same hardware path. On CUDA without
OptiX, override them with `--pipeline auto` or `--pipeline rt_fallback`.
`--require-hardware-rt` rejects software traversal. Startup logs identify the
resolved implementation and profiles contain `optix_hardware_traversal=1` for
hardware frames. Switching between hardware and software resets accumulation.

CUDA reports `DeviceRayTracingSupport()` after successful OptiX initialization;
physical-device enumeration also probes OptiX for hardware-only device selection.
Inline `RayQuery` support remains false. This integration implements Sparkium's
shared path tracer; the general graphics `CreateRayTracingProgram` API, GUI,
rasterization and presentation are not implemented on the native CUDA backend.

## Implementation and limits

Ordinary preprocessing, light CDF, film and tone-mapping kernels execute with
CUDA. Slang compiles the path tracer plus small closest-hit/miss entry points
into one CUDA source module; NVRTC emits PTX, which OptiX links and launches.
The hit/miss programs return distance, barycentrics, instance ID and primitive
ID to the shared iterative path tracer. Trace depth is one, and OptiX stack
sizes are calculated from the linked program groups.

Triangle GAS objects are cached with geometry. IAS updates rebuild when the
instance list, transforms, masks or handles change, and skip unchanged lists.
Empty scenes skip tracing a null handle. Transformed instances, mirrored and
nonuniform scales, and ordered transparent-shadow evaluation are retained.
The OptiX implementation supports triangle geometry and one shared hit group; procedural
AABBs, custom hit-group offsets, motion blur and denoising are not implemented.
It retains the CUDA backend's software texture sampling; hardware ray traversal
does not imply performance identical to Vulkan's complete graphics pipeline.

## Tests

```sh
build-release/test/sparkium/sparkium_native_test
```

Common native tests cover buffers, descriptors, texture addressing/filtering,
partial transfers, accumulation and reset on CPU/CUDA. The OptiX test
also exercises actual hardware hits, nearest-hit selection, barycentrics,
instance IDs, mirrored/nonuniform transforms, masks, invalid ranges, instance
updates, and transitions between empty and populated IAS objects. The hardware test skips when OptiX was not compiled, but runtime initialization
failures are failures. Common tests also run with OptiX disabled to verify auto
fallback and rejection of explicit hardware requests.
