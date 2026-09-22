#include "realtime/surface.hlsli"

struct VertexOutput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 barycentric : TEXCOORD0;
  [[vk::location(1)]] nointerpolation uint instance : TEXCOORD1;
  [[vk::location(2)]] nointerpolation uint primitive : TEXCOORD2;
};

VertexOutput VSMain(uint vertex : SV_VertexID) {
  ByteAddressBuffer ranges = scene_buffers[1];
  uint lo = 0, hi = ranges.Load(0);
  while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (vertex >= ranges.Load(4 + mid * 4))
      lo = mid + 1;
    else
      hi = mid;
  }
  uint instance_id = lo;
  vertex -= lo > 0 ? ranges.Load(lo * 4) : 0;
  SoftwareInstance instance = LoadSoftwareInstance(instances, instance_id);
  VertexOutput output;
  output.position = mul(parameters.view_projection, float4(VertexPosition(instance, vertex), 1));
  output.barycentric = vertex % 3 == 1 ? float2(1, 0) : (vertex % 3 == 2 ? float2(0, 1) : float2(0, 0));
  output.instance = instance_id;
  output.primitive = vertex / 3;
  return output;
}

float4 PSMain(VertexOutput input) : SV_TARGET0 {
  return float4(input.instance + 1, input.primitive, input.barycentric);
}
