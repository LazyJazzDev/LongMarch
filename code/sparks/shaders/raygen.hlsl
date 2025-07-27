#include "bindings.hlsli"
#include "common.hlsli"

[shader("raygeneration")] void Main() {
  // get the pixel coordinates
  uint sample_ind = accumulated_samples[DispatchRaysIndex().xy];
  RenderContext context;
  for (int i = 0; i < 128; i++, sample_ind++) {
    {
      uint2 pixel_coords = DispatchRaysIndex().xy;
      uint2 image_size = DispatchRaysDimensions().xy;
      float2 uv = (((float2(pixel_coords) + 0.5) / float2(image_size) * 2.0) - 1.0) * float2(1, -1);
      RayGenPayload payload;
      payload.uv = uv;
      CallShader(0, payload);
      context.rd = InitRandomSeed(pixel_coords.x, pixel_coords.y, sample_ind);
      context.origin = payload.origin;
      context.direction = payload.direction;
      context.radiance = float3(0.0, 0.0, 0.0);
      context.throughput = float3(1.0, 1.0, 1.0);
    }

    for (int bounce = 0; bounce < 32; bounce++) {
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = 1e-2;
      ray.Direction = context.direction;
      ray.TMax = 1e16;
      RayPayload payload = context.payload;
      TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
      context.payload = payload;
      MaterialRegistration mat_reg = material_regs[payload.object_id];
      context.material_buffer_index = mat_reg.buffer_index;
      if (payload.object_id >= 0) {
        // call the material shader
        CallShader(mat_reg.shader_index, context);
      } else {
        // if no object was hit, set the background color
        break;
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

    accumulated_color[DispatchRaysIndex().xy] += float4(context.radiance, 1.0);
    accumulated_samples[DispatchRaysIndex().xy] += 1;
  }
}

    // miss shader
    [shader("miss")] void MissMain(inout RayPayload payload) {
  payload.t = -1;
  payload.position = float3(0.0, 0.0, 0.0);
  payload.normal = float3(0.0, 0.0, 0.0);
  payload.tangent = float3(0.0, 0.0, 1.0);
  payload.tex_coord = float2(0.0, 0.0);
  payload.signal = 1.0;
  payload.object_id = -1;
  payload.front_facing = true;
}
