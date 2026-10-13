# RHI device creation demo

Build the `demo_rhi_hello` target using the CLion Release or Debug CMake profile.
Run the executable with `metal`, `vulkan`, or `d3d12` to select a compiled backend;
with no argument, it uses the platform default (Metal on Apple platforms).

```sh
cmake-build-release/demo/rhi_hello/demo_rhi_hello metal
```

The demo enumerates devices, prints their memory and capabilities, and creates
the automatically selected device. It links the independent `grassland_rhi`
module and does not use the old `grassland::graphics` runtime.

Metal snapshots and retains the available native devices for stable IDs within
each Core. Creating a Device retains that native `MTLDevice` and its Core;
supported capabilities are exposed automatically, without a feature enable mask.
Ray Query follows `MTLDevice::supportsRaytracing`. The DXR/Vulkan-style ray tracing
pipeline and geometry shader flags are false; Metal intersectors inside compute
or render shaders do not imply those pipeline stages are available.

Metal reports host physical memory as shared/unified memory. Dedicated VRAM
capacity is unknown (zero): `recommendedMaxWorkingSetSize` is a performance
budget, not physical VRAM capacity, and is not substituted for it. Shared memory
is not additional VRAM or a guarantee that the application can allocate it all.

This demo tests enumeration and device creation, not rendering or ray tracing.
The Metal native tests also execute and verify a buffer fill after releasing the
calling Core and a sibling Device, using Metal directly. This is an ownership
test, not an implementation of RHI command submission.
