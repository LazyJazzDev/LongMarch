#include "bindings.hlsli"
#include "common.hlsli"
#include "surface/lambertian/main.hlsl"

// sample bsdf

[shader("raygeneration")] void Main() {
  // get the pixel coordinates
  uint sample_ind = accumulated_samples[DispatchRaysIndex().xy];
  RenderContext context;
  for (int i = 0; i < scene_settings.samples_per_dispatch; i++, sample_ind++) {
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

    for (int bounce = 0; bounce < scene_settings.max_bounces; bounce++) {
      RayDesc ray;
      ray.Origin = context.origin;
      ray.TMin = T_MIN * length(context.origin);
      ray.Direction = context.direction;
      ray.TMax = 1e16;
      HitRecord hit_record = context.hit_record;
      TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, hit_record);
      context.hit_record = hit_record;
      InstanceMetadata instance_meta = instance_metadatas[context.hit_record.object_index];
      if (context.hit_record.object_index >= 0) {
        // call the surface shader
        CallShader(instance_meta.surface_shader_index, context);
        // CallableMain(context);
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

    [shader("miss")] void MissMain(inout HitRecord hit_record) {
  hit_record.t = -1;
  hit_record.position = float3(0.0, 0.0, 0.0);
  hit_record.normal = float3(0.0, 0.0, 0.0);
  hit_record.tangent = float3(0.0, 0.0, 1.0);
  hit_record.tex_coord = float2(0.0, 0.0);
  hit_record.signal = 1.0;
  hit_record.object_index = -1;
  hit_record.primitive_index = -1;
  hit_record.geom_normal = float3(0.0, 0.0, 0.0);
  hit_record.pdf = 0.0;
  // set the front facing flag to true by default
  hit_record.front_facing = true;
}
