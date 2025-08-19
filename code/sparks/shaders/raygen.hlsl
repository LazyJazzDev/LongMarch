#include "bindings.hlsli"
#include "common.hlsli"
#include "random.hlsli"

[shader("raygeneration")] void Main() {
  // get the pixel coordinates
  uint sample_ind = render_settings.accumulated_samples;
  RenderContext context;
  for (int i = 0; i < render_settings.samples_per_dispatch; i++, sample_ind++) {
    {
      uint2 pixel_coords = DispatchRaysIndex().xy;
      uint2 image_size = DispatchRaysDimensions().xy;
      context.rd = InitRandomSeed(pixel_coords.x, pixel_coords.y, sample_ind);
      float2 uv = (((float2(pixel_coords) + float2(RandomFloat(context.rd), RandomFloat(context.rd))) /
                    float2(image_size) * 2.0) -
                   1.0) *
                  float2(1, -1);
      RayGenPayload payload;
      payload.uv = uv;
      CallShader(0, payload);
      context.origin = payload.origin;
      context.direction = payload.direction;
      context.radiance = float3(0.0, 0.0, 0.0);
      context.throughput = float3(1.0, 1.0, 1.0);
      context.bsdf_pdf = INF;
    }

    for (int bounce = 0; bounce < render_settings.max_bounces; bounce++) {
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = T_MIN * max(length(context.origin), 1.0);
      ray.Direction = context.direction;
      ray.TMax = T_MAX;
      TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, context);

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

    accumulated_color[DispatchRaysIndex().xy] *= render_settings.persistence;
    accumulated_samples[DispatchRaysIndex().xy] *= render_settings.persistence;
    accumulated_color[DispatchRaysIndex().xy] += float4(context.radiance, 1.0);
    accumulated_samples[DispatchRaysIndex().xy] += 1.0;
  }
}

    [shader("miss")] void MissMain(inout RenderContext context) {
  context.throughput = float3(0.0, 0.0, 0.0);
}

[shader("miss")] void ShadowMiss(inout ShadowRayPayload payload) {
  payload.shadow = 1.0f;  // light is blocked
}
