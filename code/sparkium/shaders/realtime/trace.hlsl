#include "software/render.hlsl"
#include "realtime/parameters.hlsli"

RealtimeParameters LoadParameters() {
  ByteAddressBuffer buffer = data_buffers[SOFTWARE_DATA_BUFFER_COUNT + 6];
  RealtimeParameters p;
  p.view_projection = LoadFloat4x4(buffer, 0);
  p.previous_view_projection = LoadFloat4x4(buffer, 64);
  p.extent = buffer.Load4(128);
  p.config = buffer.Load4(144);
  p.camera_position = asfloat(buffer.Load4(160));
  return p;
}

[numthreads(8, 8, 1)] void Main(uint3 id
                                : SV_DispatchThreadID) {
  RealtimeParameters realtime = LoadParameters();
  uint period = max(1u, realtime.config.w >> 8);
  uint2 pixel = uint2(id.x * period + (id.y * 3 + realtime.config.x) % period, id.y);
  if (any(pixel >= realtime.extent.zw))
    return;
  uint2 full_pixel = min(pixel * realtime.config.z + realtime.config.z / 2, realtime.extent.xy - 1);
  float4 visibility = history_inputs[0].Load(int3(full_pixel, 0));
  float2 uv = (float2(full_pixel) + 0.5) / realtime.extent.xy * 2 - 1;
  uv.y = -uv.y;
  RenderContext context = (RenderContext)0;
  context.rd = InitRandomSeed(full_pixel.x, full_pixel.y, realtime.config.x);
  context.origin = realtime.camera_position.xyz;
  context.direction =
      normalize(mul(LoadFloat4x4(camera_data, 64), float4(uv * LoadFloat2(camera_data, 128), -1, 0)).xyz);
  context.throughput = float3(1, 1, 1);
  context.bsdf_pdf = INF;
  context.ray_type = RAY_TYPE_CAMERA;
  context.medium_object_index = -1;
  context.medium_ior = 1;
  context.medium_sample_distance = INF;
  float3 position = float3(0, 0, 0), normal = float3(0, 0, 0);
  if (visibility.x == 0) {
    context.radiance = render_settings.background_color;
  } else {
    SoftwareHit first;
    first.distance = 0;
    first.instance = uint(visibility.x) - 1;
    first.primitive = uint(visibility.y);
    first.barycentric = visibility.zw;
    HitRecord record = SoftwareHitRecord(first, context.direction);
    record.t = length(record.position - context.origin);
    position = record.position;
    normal = record.geom_normal;
    for (int bounce = 0; bounce < render_settings.max_bounces; ++bounce) {
      context.bounce = bounce;
      context.shadow_eval = float3(0, 0, 0);
      context.shadow_length = 0;
      context.shadow_dir = float3(0, 0, 0);
      if (bounce == 0) {
        SoftwareSampleMaterial(LoadSoftwareInstance(software_instances, first.instance).material, context, record);
      } else {
        RayDesc ray;
        ray.Origin = context.origin;
        ray.Direction = context.direction;
        ray.TMin = T_MIN * max(length(context.origin), 1.0);
        ray.TMax = T_MAX;
        SoftwareTracePath(ray, context);
      }
      if (any(context.shadow_eval > 0)) {
        float visibility_shadow = render_settings.alpha_shadow
                                      ? ShadowRay(context.origin, context.shadow_dir, context.shadow_length)
                                      : ShadowRayNoAlpha(context.origin, context.shadow_dir, context.shadow_length);
        context.radiance += context.shadow_eval * visibility_shadow;
      }
      if (max(context.throughput.x, max(context.throughput.y, context.throughput.z)) <= 0)
        break;
    }
  }
  float3 current = max(context.radiance, 0);
  if (!all(isfinite(current)))
    current = float3(0, 0, 0);
  current *= render_settings.clamping / max(render_settings.clamping, max(current.x, max(current.y, current.z)));
  float4 previous = accumulated_color[pixel];
  float count = min(previous.w + 1, float(realtime.config.w & 255));
  if (previous.w > 0)
    current = lerp(previous.xyz, current, 1.0 / max(count, 1));
  accumulated_color[pixel] = float4(current, count);
  realtime_outputs[1][pixel] = float4(position, visibility.x);
  realtime_outputs[2][pixel] = float4(normal, 1);
}

    [numthreads(8, 8, 1)] void Reproject(uint3 id
                                         : SV_DispatchThreadID) {
  RealtimeParameters realtime = LoadParameters();
  uint2 pixel = id.xy;
  if (any(pixel >= realtime.extent.zw))
    return;
  uint2 full_pixel = min(pixel * realtime.config.z + realtime.config.z / 2, realtime.extent.xy - 1);
  float4 visibility = history_inputs[0].Load(int3(full_pixel, 0));
  float3 position = float3(0, 0, 0), normal = float3(0, 0, 0);
  float4 previous = float4(0, 0, 0, 0);
  if (visibility.x == 0) {
    previous = float4(render_settings.background_color, 1);
  } else {
    SoftwareHit hit;
    hit.instance = uint(visibility.x) - 1;
    hit.primitive = uint(visibility.y);
    hit.barycentric = visibility.zw;
    hit.distance = 0;
    float2 uv = (float2(full_pixel) + 0.5) / realtime.extent.xy * 2 - 1;
    uv.y = -uv.y;
    float3 direction =
        normalize(mul(LoadFloat4x4(camera_data, 64), float4(uv * LoadFloat2(camera_data, 128), -1, 0)).xyz);
    HitRecord record = SoftwareHitRecord(hit, direction);
    position = record.position;
    normal = record.geom_normal;
    if (realtime.config.y) {
      float4 clip = mul(realtime.previous_view_projection, float4(position, 1));
      float2 previous_uv = clip.xy / max(clip.w, 1e-8) * float2(0.5, -0.5) + 0.5;
      if (clip.w > 0 && all(previous_uv >= 0) && all(previous_uv < 1)) {
        int2 q = min(int2(previous_uv * realtime.extent.zw), int2(realtime.extent.zw) - 1);
        float4 geometry = history_inputs[2].Load(int3(q, 0));
        float3 n = history_inputs[3].Load(int3(q, 0)).xyz;
        float pixel_size = max(1e-5, length(position - realtime.camera_position.xyz) * LoadFloat2(camera_data, 128).y *
                                         2 / realtime.extent.w);
        if (geometry.w == visibility.x && dot(n, normal) > 0.9 && length(geometry.xyz - position) < pixel_size * 2)
          previous = history_inputs[1].Load(int3(q, 0));
      }
    }
  }
  accumulated_color[pixel] = previous;
  realtime_outputs[1][pixel] = float4(position, visibility.x);
  realtime_outputs[2][pixel] = float4(normal, 1);
}
