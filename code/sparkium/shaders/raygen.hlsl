#include "bindings.hlsli"
#include "common.hlsli"
#include "random.hlsli"
#include "shadow_ray.hlsli"

void RenderPixel(uint2 pixel, uint2 extent) {
  float4 accum_color = accumulated_color[pixel];
  float accum_samples = accumulated_samples[pixel];

  // get the pixel coordinates
  uint sample_ind = render_settings.accumulated_samples;
  RenderContext context;
  for (int i = 0; i < render_settings.samples_per_dispatch; i++, sample_ind++) {
    {
      uint2 pixel_coords = pixel;
      uint2 image_size = extent;
      context.rd = InitRandomSeed(pixel_coords.x, pixel_coords.y, sample_ind);
      float2 uv = (((float2(pixel_coords) + float2(RandomFloat(context.rd), RandomFloat(context.rd))) /
                    float2(image_size) * 2.0) -
                   1.0) *
                  float2(1, -1);
      RayGenPayload payload;
      payload.uv = uv;
      payload.lens_sample = float2(RandomFloat(context.rd), RandomFloat(context.rd));
#ifdef SPARKIUM_SOFTWARE_RT
      CameraPinhole(payload);
#else
      CallShader(0, payload);
#endif
      context.origin = payload.origin;
      context.direction = payload.direction;
      context.radiance = float3(0.0, 0.0, 0.0);
      context.throughput = float3(1.0, 1.0, 1.0);
      context.bsdf_pdf = INF;
      context.ray_type = RAY_TYPE_CAMERA;
      context.medium_object_index = -1;
      context.medium_channel = 0;
      context.medium_sigma_t = float3(0.0f, 0.0f, 0.0f);
      context.medium_albedo = float3(0.0f, 0.0f, 0.0f);
      context.medium_ior = 1.0f;
      context.medium_sample_distance = INF;
    }

    for (int bounce = 0; bounce < render_settings.max_bounces; bounce++) {
      context.bounce = bounce;
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = T_MIN * max(length(context.origin), 1.0);
      ray.Direction = context.direction;
      ray.TMax = T_MAX;

      context.shadow_eval = float3(0.0, 0.0, 0.0);
      context.shadow_dir = float3(0.0, 0.0, 0.0);
      context.shadow_length = 0.0;

#ifdef SPARKIUM_SOFTWARE_RT
      SoftwareTracePath(ray, context);
#else
      TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, context);
#endif

      if (context.shadow_eval.x > 0.0 || context.shadow_eval.y > 0.0 || context.shadow_eval.z > 0.0) {
        if (render_settings.alpha_shadow) {
          context.radiance +=
              context.shadow_eval * ShadowRay(context.origin, context.shadow_dir, context.shadow_length);
        } else {
          context.radiance +=
              context.shadow_eval * ShadowRayNoAlpha(context.origin, context.shadow_dir, context.shadow_length);
        }
      }

      // russian roulette
      float p = max(max(context.throughput.x, context.throughput.y), context.throughput.z);
      if (p < 1.0) {
        float r = RandomFloat(context.rd);
        if (r >= p) {
          break;  // terminate the path
        } else {
          context.throughput /= p;  // continue the path
        }
      }
    }

    context.radiance *= render_settings.clamping / max(max(render_settings.clamping, context.radiance.r),
                                                       max(context.radiance.g, context.radiance.b));
    accum_color *= render_settings.persistence;
    accum_samples *= render_settings.persistence;
    accum_color += float4(context.radiance, 1.0);
    accum_samples += 1.0;
    float exposure_clamping = accum_samples * render_settings.max_exposure;
    accum_color.rgb *=
        exposure_clamping / max(max(exposure_clamping, accum_color.r), max(accum_color.g, accum_color.b));
  }

  accumulated_color[pixel] = accum_color;
  accumulated_samples[pixel] = accum_samples;
}

void ApplyPathMiss(inout RenderContext context) {
  if (context.medium_object_index >= 0) {
    context.throughput = float3(0.0f, 0.0f, 0.0f);
    return;
  }
  context.radiance += render_settings.background_color * context.throughput;
  context.throughput = float3(0.0, 0.0, 0.0);
}

#ifndef SPARKIUM_SOFTWARE_RT
[shader("raygeneration")] void Main() { RenderPixel(DispatchRaysIndex().xy, DispatchRaysDimensions().xy); }

    [shader("miss")] void MissMain(inout RenderContext context) {
  ApplyPathMiss(context);
}

[shader("miss")] void ShadowMiss(inout ShadowRayPayload payload) {
  // Keep transmittance accumulated by transparent any-hit shaders.
}

#endif
