struct DisplaySettings {
  float headroom;
  float3 padding;
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

float4 PSMain(Varying input) : SV_Target0 {
  float3 color = radiance.Load(int3(int2(input.position.xy), 0)).rgb;
  // Match nbody_cs: dense overlap approaches white while preserving luminance.
  float peak = max(color.r, max(color.g, color.b));
  float highlight = 1.0 - rcp(max(peak, 1.0));
  float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
  color = lerp(color, luminance.xxx, highlight);
  peak = max(color.r, max(color.g, color.b));
  color *= min(1.0, display.headroom / max(peak, 1e-6));
  return float4(color, 1);
}
