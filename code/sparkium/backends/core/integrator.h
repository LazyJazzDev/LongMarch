#pragma once
// Portable port of code/sparkium/shaders/raygen.hlsl (RenderPixel and
// ApplyPathMiss) including the software path of software/render.hlsl
// (SoftwareTracePath). The byte-address buffers and the hardware/callable
// dispatch became typed scene access; the sample ordering, the Russian
// roulette, the clamping and the film accumulation are unchanged.

#include "sparkium/backends/core/camera.h"
#include "sparkium/backends/core/geometry.h"
#include "sparkium/backends/core/materials.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline void ApplyPathMiss(const RenderSettings &settings, RenderContext &context) {
  context.radiance += settings.background_color * context.throughput;
  context.throughput = float3(0.0f, 0.0f, 0.0f);
}

// One pixel, `settings.samples_per_dispatch` samples, accumulating into the
// film's per-pixel color/sample counters exactly like the HLSL raygen shader.
SPARKIUM_HD inline void RenderPixel(const DeviceScene &scene,
                                    const RenderSettings &settings,
                                    uint32_t pixel_x,
                                    uint32_t pixel_y,
                                    uint32_t extent_x,
                                    uint32_t extent_y,
                                    float4 *accumulated_color,
                                    float *accumulated_samples) {
  float4 accum_color = *accumulated_color;
  float accum_samples = *accumulated_samples;

  uint32_t sample_ind = static_cast<uint32_t>(settings.accumulated_samples);
  RenderContext context;
  for (int32_t i = 0; i < settings.samples_per_dispatch; i++, sample_ind++) {
    {
      // HLSL spells this float2(pixel) / float2(image_size).
      float2 pixel_coords = float2(static_cast<float>(pixel_x), static_cast<float>(pixel_y));
      float2 image_size = float2(static_cast<float>(extent_x), static_cast<float>(extent_y));
      context.rd = InitRandomSeed(pixel_x, pixel_y, sample_ind, scene.sobol_table);
      // The two jitter samples are consumed in order, like the two
      // RandomFloat calls of raygen.hlsl (see RandomFloat2).
      float2 jitter = RandomFloat2(context.rd, scene.sobol_table);
      float2 uv = (((pixel_coords + jitter) / image_size * 2.0f) - 1.0f) * float2(1.0f, -1.0f);
      RayGenPayload payload;
      payload.uv = uv;
      payload.lens_sample = RandomFloat2(context.rd, scene.sobol_table);
      CameraPinhole(scene, payload);
      context.origin = payload.origin;
      context.direction = payload.direction;
      context.radiance = float3(0.0, 0.0, 0.0);
      context.throughput = float3(1.0, 1.0, 1.0);
      context.bsdf_pdf = SPARKIUM_INF;
      context.ray_type = SPARKIUM_RAY_TYPE_CAMERA;
      context.medium_object_index = -1;
    }

    for (int32_t bounce = 0; bounce < settings.max_bounces; bounce++) {
      context.bounce = bounce;
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = SPARKIUM_T_MIN * device::max(device::length(context.origin), 1.0f);
      ray.Direction = context.direction;
      ray.TMax = SPARKIUM_T_MAX;

      context.shadow_eval = float3(0.0, 0.0, 0.0);
      context.shadow_dir = float3(0.0, 0.0, 0.0);
      context.shadow_length = 0.0;

      SoftwareHit hit;
      if (!InlineIntersect(scene, ray, false, hit)) {
        ApplyPathMiss(settings, context);
      } else {
        const InstanceData &instance = scene.instances[hit.instance];
        HitRecord hit_record = MakeMeshHitRecord(instance.mesh, hit.instance, hit.primitive, hit.barycentric,
                                                 hit.distance, ray.Direction, instance.object_to_world,
                                                 instance.world_to_object, scene);
        SampleMaterial(scene, instance.material, context, hit_record);
      }

      if (context.shadow_eval.x > 0.0f || context.shadow_eval.y > 0.0f || context.shadow_eval.z > 0.0f) {
        if (settings.alpha_shadow != 0) {
          context.radiance +=
              context.shadow_eval * ShadowRay(scene, context.origin, context.shadow_dir, context.shadow_length);
        } else {
          context.radiance +=
              context.shadow_eval * ShadowRayNoAlpha(scene, context.origin, context.shadow_dir, context.shadow_length);
        }
      }

      // Russian roulette.
      float p = device::max(device::max(context.throughput.x, context.throughput.y), context.throughput.z);
      if (p < 1.0f) {
        float r = RandomFloat(context.rd, scene.sobol_table);
        if (r >= p) {
          break;  // terminate the path
        } else {
          context.throughput /= p;  // continue the path
        }
      }
    }

    context.radiance *=
        settings.clamping / device::max(device::max(settings.clamping, context.radiance.x),
                                        device::max(context.radiance.y, context.radiance.z));
    accum_color *= settings.persistence;
    accum_samples *= settings.persistence;
    accum_color = accum_color + float4(context.radiance, 1.0f);
    accum_samples += 1.0f;
    float exposure_clamping = accum_samples * settings.max_exposure;
    float3 rgb = accum_color.xyz() *
                 (exposure_clamping / device::max(device::max(exposure_clamping, accum_color.x),
                                                  device::max(accum_color.y, accum_color.z)));
    accum_color = float4(rgb, accum_color.w);
  }

  *accumulated_color = accum_color;
  *accumulated_samples = accum_samples;
}

}  // namespace sparkium::backends
