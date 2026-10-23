struct Varying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

Varying VSMain(uint id : SV_VertexID) {
  Varying o;
  o.uv = float2((id << 1) & 2, id & 2);
  o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return o;
}

float3 Encode(float3 linear709) {
#if OUTPUT_COLOR_SPACE == 1
  return linear709;  // scRGB: 1 = 80 nits.
#elif OUTPUT_COLOR_SPACE == 2
  const float3x3 to2020 =
      float3x3(0.627404, 0.329283, 0.043313, 0.069097, 0.919540, 0.011362, 0.016391, 0.088013, 0.895595);
  float3 normalized = max(mul(to2020, linear709), 0) * (80.0 / 10000.0);
  float3 p = pow(normalized, 2610.0 / 16384.0);
  return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) / (1 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
#else
  float3 c = saturate(linear709);
  return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
#endif
}

float4 PSMain(Varying input) : SV_Target0 {
  float x = saturate(input.uv.x), y = saturate(input.uv.y);
  float nits = x < 0.2 ? 20 : x < 0.4 ? 80 : x < 0.6 ? 203 : x < 0.8 ? 400 : 1000;
  float3 color = nits / 80.0;
  if (y > 0.5) {
    float3 tint = y < 0.65 ? float3(1, 0, 0) : y < 0.8 ? float3(0, 1, 0) : float3(0, 0, 1);
    color *= tint;
  }
  return float4(Encode(color), 1);
}
