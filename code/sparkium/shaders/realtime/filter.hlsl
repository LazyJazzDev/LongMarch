#include "parameters.hlsli"
ConstantBuffer<RealtimeParameters> parameters : register(b0, space0);
Texture2D<float4> inputs[] : register(t0, space1);
RWTexture2D<float4> output : register(u0, space2);
[numthreads(8, 8, 1)] void Main(uint3 id
                                : SV_DispatchThreadID) {
  uint2 pixel = id.xy;
  if (any(pixel >= parameters.extent.zw))
    return;
  float4 geometry = inputs[1].Load(int3(pixel, 0));
  float3 normal = inputs[2].Load(int3(pixel, 0)).xyz;
  float radius = max(1e-5, length(geometry.xyz - parameters.camera_position.xyz) * 6 / max(parameters.extent.w, 1u));
  float3 sum = float3(0, 0, 0);
  float total = 0;
  for (int y = -3; y <= 3; ++y)
    for (int x = -3; x <= 3; ++x) {
      int2 q = clamp(int2(pixel) + int2(x, y), int2(0, 0), int2(parameters.extent.zw) - 1);
      float4 color = inputs[0].Load(int3(q, 0));
      float4 sample_geometry = inputs[1].Load(int3(q, 0));
      float3 n = inputs[2].Load(int3(q, 0)).xyz;
      if (color.w == 0 || geometry.w != sample_geometry.w || (geometry.w > 0 && dot(normal, n) < 0.9))
        continue;
      float3 difference = sample_geometry.xyz - geometry.xyz;
      float weight = exp(-float(x * x + y * y) / 8.0);
      if (geometry.w > 0)
        weight *= exp(-dot(difference, difference) / (radius * radius) - abs(dot(normal, difference)) * 32 / radius);
      weight *= min(color.w, 4);
      sum += color.xyz * weight;
      total += weight;
    }
  output[pixel] = total > 1e-8 ? float4(sum / total, 1) : float4(0, 0, 0, 0);
}
