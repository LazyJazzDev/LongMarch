#include "realtime/surface.hlsli"
Texture2D<float4> inputs[] : register(t0, space3);
RWTexture2D<float4> output : register(u0, space4);
[numthreads(8, 8, 1)] void Main(uint3 id
                                : SV_DispatchThreadID) {
  uint2 pixel = id.xy;
  if (any(pixel >= parameters.extent.xy))
    return;
  float4 visibility = inputs[0].Load(int3(pixel, 0));
  int2 center = min(int2(pixel / parameters.config.z), int2(parameters.extent.zw) - 1);
  float3 position = float3(0, 0, 0), normal = float3(0, 0, 0);
  if (visibility.x > 0)
    Surface(visibility, position, normal);
  float radius = max(1e-5, length(position - parameters.camera_position.xyz) * 4.0 / max(parameters.extent.w, 1u));
  float3 sum = float3(0, 0, 0);
  float total = 0;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
      int2 q = clamp(center + int2(x, y), int2(0, 0), int2(parameters.extent.zw) - 1);
      float4 lighting = inputs[1].Load(int3(q, 0));
      if (lighting.w == 0)
        continue;
      float4 geometry = inputs[2].Load(int3(q, 0));
      if (geometry.w != visibility.x)
        continue;
      float3 n = inputs[3].Load(int3(q, 0)).xyz;
      if (visibility.x > 0 && dot(normal, n) < 0.85)
        continue;
      float3 difference = geometry.xyz - position;
      float plane = abs(dot(difference, normal));
      float weight = exp(-float(x * x + y * y) / 4.0);
      if (visibility.x > 0)
        weight *= exp(-dot(difference, difference) / (radius * radius) - plane * 32 / radius);
      sum += lighting.xyz * weight;
      total += weight;
    }
  float3 color = total > 1e-8 ? sum / total : inputs[1].Load(int3(center, 0)).xyz;
  output[pixel] = float4(color, 1);
}
