struct BlurSettings {
  int2 direction;
  int2 extent;
};

[[vk::binding(0, 0)]] Texture2D<float4> blur_source : register(t0, space0);
[[vk::binding(0, 1)]] RWTexture2D<float4> blur_target : register(u0, space1);
[[vk::binding(0, 2)]] ConstantBuffer<BlurSettings> blur_settings : register(b0, space2);

[numthreads(8, 8, 1)] void Blur(uint3 gid
                                : SV_DispatchThreadID) {
  int2 pixel = int2(gid.xy);
  if (any(pixel >= blur_settings.extent))
    return;
  const float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};
  float4 color = blur_source.Load(int3(pixel, 0)) * weights[0];
  for (int i = 1; i < 5; ++i) {
    int2 offset = blur_settings.direction * i;
    color += blur_source.Load(int3(clamp(pixel + offset, int2(0, 0), blur_settings.extent - 1), 0)) * weights[i];
    color += blur_source.Load(int3(clamp(pixel - offset, int2(0, 0), blur_settings.extent - 1), 0)) * weights[i];
  }
  blur_target[pixel] = color;
}
