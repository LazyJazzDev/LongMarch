// Platform presentation convention; callers may override it in the source prefix.
#ifndef PQ_NITS_PER_UNIT
#if RHI_PLATFORM_MACOS
#define PQ_NITS_PER_UNIT 100.0
#elif RHI_PLATFORM_LINUX
// Wayland PQ source reference white; output feedback must not be applied twice.
#define PQ_NITS_PER_UNIT 203.0
#else
#define PQ_NITS_PER_UNIT 80.0
#endif
#endif

struct DisplaySettings {
  float linear_scale;
  float headroom;
  float2 padding;
};

RHI_UNIFORM(0, 0, DisplaySettings, display);
RHI_TEXTURE2D(0, 1, float4, radiance);

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
  return linear709 * display.linear_scale;  // Relative white to native linear encoding.
#elif OUTPUT_COLOR_SPACE == 3
  return linear709;  // Native linear Rec.709; no scRGB absolute-white compensation.
#elif OUTPUT_COLOR_SPACE == 2
  const float3x3 to2020 =
      float3x3(0.627404, 0.329283, 0.043313, 0.069097, 0.919540, 0.011362, 0.016391, 0.088013, 0.895595);
  float3 normalized = max(mul(to2020, linear709), 0) * (display.linear_scale * (PQ_NITS_PER_UNIT / 10000.0));
  normalized /= max(1.0, max(normalized.r, max(normalized.g, normalized.b)));
  float3 p = pow(normalized, 2610.0 / 16384.0);
  return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) / (1 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
#else
  float3 c = saturate(linear709);
  return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
#endif
}

float4 PSMain(Varying input) : SV_Target0 {
  float3 color = radiance.Load(int3(int2(input.position.xy), 0)).rgb;
  // Match nbody_cs: dense overlap approaches white while preserving luminance.
  float peak = max(color.r, max(color.g, color.b));
  float highlight = 1.0 - rcp(max(peak, 1.0));
  float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
  color = lerp(color, luminance.xxx, highlight);
  if (OUTPUT_COLOR_SPACE != 0) {
    peak = max(color.r, max(color.g, color.b));
    color *= min(1.0, display.headroom / max(peak, 1e-6));
  }
  return float4(Encode(color), 1);
}
