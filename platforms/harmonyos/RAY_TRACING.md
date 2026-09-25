# HarmonyOS Vulkan ray tracing investigation

## Verified on 2026-09-25

The signed application queried the connected MLN-AL00 running
6.1.0.135(SP17C00E135R3P4). Vulkan identifies its GPU as Maleoon 935F and its
API as Vulkan 1.3. This report concerns public API availability for this app
and OS build, not the presence or absence of ray-tracing units in the silicon.

| Query | Result |
| --- | --- |
| `VK_KHR_acceleration_structure` | Not advertised |
| `VK_KHR_ray_query` | Not advertised |
| `VK_KHR_ray_tracing_pipeline` | Not advertised |
| `HMS_XEG_EnumerateDeviceExtensionProperties` | `VK_SUCCESS`, four extensions |
| XEngine extensions | `XEG_spatial_upscale` v1, `XEG_temporal_upscale` v2, `XEG_adaptive_vrs` v1, `XEG_hps_radix_sort` v1 |
| `XEG_rtgi`, `XEG_rt_reflection`, `XEG_rt_shadow_ao` | Not returned |

The enumerated Vulkan instance layers were `VK_LAYER_HUAWEI_iGraphics`,
`VK_LAYER_HUAWEI_GameAware`, and `VK_LAYER_OHOS_surface`. Their names alone do
not establish ray-tracing support or a documented way to enable it.

`native/XEngineCapabilities.cpp` optionally loads the public `libxengine.so`
and invokes the official extension query. It does not confuse XEngine names
with Vulkan device extensions, enable unadvertised features, or depend on
private driver entry points. Results appear in the `LongMarchGPU` hilog tag.
No XEngine ray-tracing operation has been executed on this device.

## What the official SDK interfaces provide

Source: Huawei SDK 26.0.0.105 installed with DevEco Studio, under
`Contents/sdk/default/hms/native/sysroot/usr/include/xengine/`.
These are actual SDK declarations and API contracts, not inferred marketing
capabilities. The extension query dates from API 12; the RT interfaces below
are marked HarmonyOS 6.0.0 / API 20. API version alone does not guarantee that a
device returns these extensions.

| Header | Contract and consequence |
| --- | --- |
| `xeg_vulkan_extension.h` | Query XEngine capabilities using `HMS_XEG_EnumerateDeviceExtensionProperties(VkPhysicalDevice, ...)`. Each RT header requires checking its extension before use. |
| `xeg_vulkan_rt_reflection.h` | Takes ray-origin/direction images and a **`VkAccelerationStructureKHR`**. Produces packed nearest-hit information (miss flag, primitive/instance/geometry IDs, barycentrics, distance). This can perform batched intersection, but does not provide a BLAS/TLAS builder or arbitrary ray-generation/hit/miss shaders. |
| `xeg_vulkan_rt_visible_mask.h` | Shadow/AO rendering also consumes a **`VkAccelerationStructureKHR`**, plus the documented G-buffer/light parameters. |
| `xeg_vulkan_rtgi.h` | DDGI consumes probe-ray directions, **already evaluated hit radiance/distance**, and hit normal/metalness, then updates probes and produces GI. NNGI consumes training GI and scene/G-buffer data for inference. These interfaces do not replace ray intersection or material/light evaluation. |
| `xeg_vulkan_common.h` | Includes command synchronization for consuming XEngine results. Integrations must also satisfy each resource's Vulkan layout/access/lifetime requirements. |

In particular, the presence of `xeg_vulkan_rtgi.h` on the development machine
is not sufficient to turn on hardware path tracing. DDGI is a GI reconstruction
component, and reflection/shadow entry points require a valid Vulkan
acceleration structure. The inspected XEngine headers expose no alternate
acceleration-structure construction API.

## Implementation route for LongMarch

### Preferred: standard Vulkan ray query

LongMarch already has Vulkan acceleration-structure creation and Sparkium's
ray-query renderer. When a device advertises the required extensions and
feature bits, use this route first:

1. Check acceleration structures, ray query, deferred host operations, and
   buffer device address (including promoted core features where applicable).
   Enable the supported features in `VkDevice` creation.
2. Build BLAS for mesh geometry and TLAS for instances; preserve the existing
   material/instance indexing. Synchronize builds before shader reads.
3. Execute the existing compute path tracer with hardware ray queries; preserve
   shader-graph materials, transmission, bounce counts and HDR accumulation.
4. Validate a single triangle, instancing, shadow rays, then Cornell, Texture
   and all Blender scenes against the existing renderer. Check hit IDs,
   barycentrics, finite radiance and stable sample accumulation before measuring
   performance.

A separate `VK_KHR_ray_tracing_pipeline` implementation uses shader groups,
shader binding tables and `vkCmdTraceRaysKHR`; it is a distinct capability.
Ray-query support can accelerate tracing without exposing that full pipeline.
Do not require the full pipeline merely to select ray query.

### Optional: XEngine reflection as an intersection stage

Only attempt this when `XEG_rt_reflection` is returned **and** a supported way
to construct its `VkAccelerationStructureKHR` is available. Split tracing into
passes: generate rays → XEngine intersection → decode hit records → evaluate
materials/lights → emit subsequent/shadow rays → accumulate. Integrate
`HMS_XEG_CreateRTReflection`, `HMS_XEG_CmdRenderRTReflection` and destruction
with the engine's command/resource lifetime model.

This is a renderer change, not a replacement call for `vkCmdTraceRaysKHR`.
Validate packed ID limits, material mapping, culling, alpha/transmission and
shadow semantics before claiming feature parity. Nearest-hit output alone
is not a substitute for all existing any-hit/transparency behavior.

### Optional: DDGI/NNGI hybrid rendering

If `XEG_rtgi` is available, add an explicitly selected approximate real-time GI
mode. Supply G-buffers and traced/shaded probe rays for DDGI, or the required
training GI for NNGI; call the documented create/render/synchronization APIs.
This can complement hardware ray query or another ray generator. It should
not silently replace the reference path tracer because reconstruction and
training change rendering behavior and image equivalence.

## Current gate and next evidence needed

Both the standard and XEngine RT capability checks are negative on the tested
OS build. Keep the compute fallback active. To proceed with actual hardware
RT, obtain a device/driver that returns the required extensions, or a documented
Huawei developer enablement procedure and supported-device matrix. Whether
this model requires a later OS, a different device, or vendor enablement is
not established by the available SDK headers or this query. Do not invent a
permission, whitelist or layer setting as a remedy.

Retest both capability sets after any OS/device change; then run the minimal
intersection validation before integrating full scenes. Improving current
compute compilation and dispatch remains useful independently of that gate.
