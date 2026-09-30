struct GlassSettings {
  int2 extent;
  int count;
  int pad;
  float4 rects[16];
};

[[vk::binding(0, 0)]] Texture2D<float4> scene_source : register(t0, space0);
[[vk::binding(0, 1)]] Texture2D<float4> blurred_source : register(t0, space1);
[[vk::binding(0, 2)]] RWTexture2D<float4> glass_target : register(u0, space2);
[[vk::binding(0, 3)]] ConstantBuffer<GlassSettings> glass_settings : register(b0, space3);

[numthreads(8, 8, 1)] void Composite(uint3 gid
                                     : SV_DispatchThreadID) {
  int2 pixel = int2(gid.xy);
  if (any(pixel >= glass_settings.extent))
    return;
  float4 scene = scene_source.Load(int3(pixel, 0));
  float4 blur = blurred_source.Load(int3(pixel, 0));
  float amount = 0.0f;
  for (int i = 0; i < glass_settings.count; ++i) {
    float4 rect = glass_settings.rects[i];
    float2 center = (rect.xy + rect.zw) * 0.5f;
    float2 half_size = (rect.zw - rect.xy) * 0.5f;
    float radius = min(16.0f, min(half_size.x, half_size.y));
    float2 delta = abs(float2(pixel) + 0.5f - center) - (half_size - radius);
    float sdf = length(max(delta, float2(0, 0))) + min(max(delta.x, delta.y), 0.0f) - radius;
    amount = max(amount, saturate(0.5f - sdf));
  }
  glass_target[pixel] = lerp(scene, blur, amount * 0.72f);
}
