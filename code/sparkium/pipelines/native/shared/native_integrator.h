#pragma once

// Port of `camera.hlsl`, `raygen.hlsl` and `software/{render.hlsl,shadow.hlsli}`.
// This is the whole per-pixel integrator; the CPU and CUDA backends only
// supply the pixel loop around `RenderPixel`.

#include "native_materials.h"
#include "native_traversal.h"

namespace sparkium::native {

struct RayGenPayload {
  float2 uv;
  float2 lens_sample;
  float3 origin;
  float3 direction;
};

// ---------------------------------------------------------------------------
// camera.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void CameraPinhole(const SceneView &scene, RayGenPayload &raygen_payload) {
  const glm::mat4 camera_to_world = LoadFloat4x4(scene.camera_data, 64);
  const float2 scale = LoadFloat2(scene.camera_data, 128);
  const float aperture_radius = LoadFloat(scene.camera_data, 136);
  const float focus_distance = LoadFloat(scene.camera_data, 140);
  const int aperture_blades = static_cast<int>(scene.camera_data.Load(144));
  const float aperture_rotation = LoadFloat(scene.camera_data, 148);
  const float aperture_ratio = LoadFloat(scene.camera_data, 152);

  raygen_payload.origin = float3{0, 0, 0};
  raygen_payload.direction = glm::normalize(float3{raygen_payload.uv * scale, -1.0f});
  if (aperture_radius > 0.0f && focus_distance > 0.0f) {
    float2 aperture_sample;
    if (aperture_blades >= 3) {
      const float sector_position = raygen_payload.lens_sample.x * static_cast<float>(aperture_blades);
      const int sector = glm::min(static_cast<int>(sector_position), aperture_blades - 1);
      const float edge_factor = frac(sector_position);
      const float angle0 = aperture_rotation + 2.0f * PI * static_cast<float>(sector) / aperture_blades;
      const float angle1 = aperture_rotation + 2.0f * PI * static_cast<float>(sector + 1) / aperture_blades;
      const float2 vertex0{::cosf(angle0), ::sinf(angle0)};
      const float2 vertex1{::cosf(angle1), ::sinf(angle1)};
      aperture_sample = ::sqrtf(raygen_payload.lens_sample.y) *
                        float2{lerp(vertex0.x, vertex1.x, edge_factor), lerp(vertex0.y, vertex1.y, edge_factor)};
    } else {
      const float radius = ::sqrtf(raygen_payload.lens_sample.x);
      const float angle = 2.0f * PI * raygen_payload.lens_sample.y;
      aperture_sample = radius * float2{::cosf(angle), ::sinf(angle)};
    }
    aperture_sample.x *= ::fmaxf(aperture_ratio, 1e-6f);
    const float3 lens_position{aperture_radius * aperture_sample, 0.0f};
    const float3 focus_position = raygen_payload.direction * (focus_distance / -raygen_payload.direction.z);
    raygen_payload.origin = lens_position;
    raygen_payload.direction = glm::normalize(focus_position - lens_position);
  }
  raygen_payload.origin = float3{camera_to_world * float4{raygen_payload.origin, 1.0f}};
  raygen_payload.direction = glm::normalize(float3{camera_to_world * float4{raygen_payload.direction, 0.0f}});
}

// ---------------------------------------------------------------------------
// software/render.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline HitRecord SoftwareHitRecord(const SceneView &scene,
                                                  const SoftwareHit &hit,
                                                  const float3 &direction) {
  const SoftwareInstance instance = LoadSoftwareInstance(scene.software_instances, hit.instance);
  // The HLSL renderer passes `transpose(instance.world_to_object)`; the port
  // takes the float3x4 directly and multiplies from the right.
  return MakeMeshHitRecord(scene, instance.geometry, hit.instance, hit.primitive, hit.barycentric, hit.distance,
                           direction, instance.object_to_world, instance.world_to_object);
}

// ---------------------------------------------------------------------------
// software/shadow.hlsli
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float ShadowRayNoAlpha(const SceneView &scene,
                                             const float3 &origin,
                                             const float3 &direction,
                                             float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * ::fmaxf(glm::length(origin), 1.0f);
  ray.TMax = dist;
  SoftwareHit hit;
  return InlineIntersect(scene, ray, true, hit) ? 0.0f : 1.0f;
}

LM_DEVICE_FUNC inline float ShadowRay(const SceneView &scene,
                                      const float3 &origin,
                                      const float3 &direction,
                                      float dist) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * ::fmaxf(glm::length(origin), 1.0f);
  ray.TMax = dist;
  float transmission = 1.0f;
  SoftwareHit hit;
  while (transmission > 1.0e-4f && InlineIntersect(scene, ray, false, hit)) {
    const uint32_t material = LoadSoftwareInstance(scene.software_instances, hit.instance).material;
    transmission *= ShadowTransmission(scene, material, SoftwareHitRecord(scene, hit, direction), direction);
    // Advance one representable positive ray parameter, preserving close transparent layers.
    ray.TMin = asfloat(asuint(hit.distance) + 1);
  }
  return transmission;
}

// ---------------------------------------------------------------------------
// raygen.hlsl
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void ApplyPathMiss(const SceneView &scene, RenderContext &context) {
  if (context.medium_object_index >= 0) {
    context.throughput = float3{0.0f, 0.0f, 0.0f};
    return;
  }
  context.radiance += scene.settings.background_color * context.throughput;
  context.throughput = float3{0.0f, 0.0f, 0.0f};
}

LM_DEVICE_FUNC inline void SoftwareTracePath(const SceneView &scene, const RayDesc &ray, RenderContext &context) {
  SoftwareHit hit;
  if (!InlineIntersect(scene, ray, false, hit)) {
    ApplyPathMiss(scene, context);
    return;
  }
  const HitRecord record = SoftwareHitRecord(scene, hit, ray.Direction);
  if (!ContinueSubsurfaceRandomWalk(scene, context, record))
    SampleMaterial(scene, LoadSoftwareInstance(scene.software_instances, hit.instance).material, context, record);
}

// Renders `settings.samples_per_dispatch` samples into the accumulation
// buffers, exactly as the compute `Main` entry point does per thread.
LM_DEVICE_FUNC inline void RenderPixel(const SceneView &scene,
                                       uint2 pixel,
                                       uint2 extent,
                                       float4 *accumulated_color,
                                       float *accumulated_samples) {
  const uint32_t pixel_index = pixel.y * extent.x + pixel.x;
  float4 accum_color = accumulated_color[pixel_index];
  float accum_samples = accumulated_samples[pixel_index];

  // get the pixel coordinates
  uint32_t sample_ind = static_cast<uint32_t>(scene.settings.accumulated_samples);
  RenderContext context;
  for (int i = 0; i < scene.settings.samples_per_dispatch; i++, sample_ind++) {
    {
      context.rd = InitRandomSeed(pixel.x, pixel.y, sample_ind);
      const float2 uv = ((float2{static_cast<float>(pixel.x), static_cast<float>(pixel.y)} +
                          float2{RandomFloat(scene, context.rd), RandomFloat(scene, context.rd)}) /
                             float2{static_cast<float>(extent.x), static_cast<float>(extent.y)} * 2.0f -
                         1.0f) *
                        float2{1.0f, -1.0f};
      RayGenPayload payload;
      payload.uv = uv;
      payload.lens_sample = float2{RandomFloat(scene, context.rd), RandomFloat(scene, context.rd)};
      CameraPinhole(scene, payload);
      context.origin = payload.origin;
      context.direction = payload.direction;
      context.radiance = float3{0.0f, 0.0f, 0.0f};
      context.throughput = float3{1.0f, 1.0f, 1.0f};
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_CAMERA;
      context.medium_object_index = -1;
      context.medium_channel = 0;
      context.medium_sigma_t = float3{0.0f, 0.0f, 0.0f};
      context.medium_albedo = float3{0.0f, 0.0f, 0.0f};
      context.medium_ior = 1.0f;
      context.medium_sample_distance = INF;
    }

    for (int bounce = 0; bounce < scene.settings.max_bounces; bounce++) {
      context.bounce = bounce;
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = T_MIN * ::fmaxf(glm::length(context.origin), 1.0f);
      ray.Direction = context.direction;
      ray.TMax = T_MAX;

      context.shadow_eval = float3{0.0f, 0.0f, 0.0f};
      context.shadow_dir = float3{0.0f, 0.0f, 0.0f};
      context.shadow_length = 0.0f;

      SoftwareTracePath(scene, ray, context);

      if (context.shadow_eval.x > 0.0f || context.shadow_eval.y > 0.0f || context.shadow_eval.z > 0.0f) {
        if (scene.settings.alpha_shadow) {
          context.radiance +=
              context.shadow_eval * ShadowRay(scene, context.origin, context.shadow_dir, context.shadow_length);
        } else {
          context.radiance +=
              context.shadow_eval * ShadowRayNoAlpha(scene, context.origin, context.shadow_dir, context.shadow_length);
        }
      }

      // russian roulette
      const float p = ::fmaxf(::fmaxf(context.throughput.x, context.throughput.y), context.throughput.z);
      if (p < 1.0f) {
        const float r = RandomFloat(scene, context.rd);
        if (r >= p) {
          break;  // terminate the path
        } else {
          context.throughput /= p;  // continue the path
        }
      }
    }

    const float clamping = scene.settings.clamping;
    context.radiance *= clamping / ::fmaxf(::fmaxf(clamping, context.radiance.x),
                                           ::fmaxf(context.radiance.y, context.radiance.z));
    accum_color *= scene.settings.persistence;
    accum_samples *= scene.settings.persistence;
    accum_color += float4{context.radiance, 1.0f};
    accum_samples += 1.0f;
    const float exposure_clamping = accum_samples * scene.settings.max_exposure;
    const float3 clamped = float3{accum_color.x, accum_color.y, accum_color.z} *
                           (exposure_clamping / ::fmaxf(::fmaxf(exposure_clamping, accum_color.x),
                                                        ::fmaxf(accum_color.y, accum_color.z)));
    accum_color.x = clamped.x;
    accum_color.y = clamped.y;
    accum_color.z = clamped.z;
  }

  accumulated_color[pixel_index] = accum_color;
  accumulated_samples[pixel_index] = accum_samples;
}

// ---------------------------------------------------------------------------
// film2img.hlsl and tone_mapping.hlsl
// ---------------------------------------------------------------------------
// `film2img.hlsl` reads the sample count through an `int`, so a fractional
// count left by film persistence truncates the same way here.
LM_DEVICE_FUNC inline float4 FilmToImage(const float4 &color, float samples) {
  const int sample_count = static_cast<int>(samples);
  if (sample_count == 0)
    return float4{0.0f, 0.0f, 0.0f, 1.0f};
  return color / static_cast<float>(sample_count);
}

LM_DEVICE_FUNC inline float Linear2sRGB(float value) {
  // step(0.0031308, value) selects the higher branch at exactly the cutoff.
  return value < 0.0031308f ? value * 12.92f : 1.055f * ::powf(value, 1.0f / 2.4f) - 0.055f;
}

LM_DEVICE_FUNC inline float3 Linear2sRGB(const float3 &color) {
  return float3{Linear2sRGB(color.x), Linear2sRGB(color.y), Linear2sRGB(color.z)};
}

// Smooth shoulder/toe approximation for legacy Blender Filmic scenes.
LM_DEVICE_FUNC inline float3 FilmicCurve(float3 color) {
  color = glm::max(color, make_float3(0.0f));
  return saturate((color * (2.51f * color + 0.03f)) / (color * (2.43f * color + 0.59f) + 0.14f));
}

LM_DEVICE_FUNC inline float4 ToneMap(const RenderSettings &settings, const float4 &color) {
  const float3 linear_color =
      glm::max(float3{color.x, color.y, color.z} * ::exp2f(settings.exposure), make_float3(0.0f));
  float3 mapped;
  if (settings.view_transform == 1) {
    mapped = saturate(Linear2sRGB(linear_color));
  } else if (settings.view_transform == 2) {
    mapped = FilmicCurve(linear_color);
    mapped = saturate((mapped - 0.18f) * settings.contrast + 0.18f);
    const float inv_gamma = 1.0f / ::fmaxf(settings.gamma, 1e-4f);
    mapped = float3{::powf(mapped.x, inv_gamma), ::powf(mapped.y, inv_gamma), ::powf(mapped.z, inv_gamma)};
  } else {
    const float max_channel = ::fmaxf(::fmaxf(linear_color.x, linear_color.y), linear_color.z);
    mapped = Linear2sRGB(linear_color / ::fmaxf(1.0f, max_channel));
  }
  return float4{mapped, color.w};
}

}  // namespace sparkium::native
