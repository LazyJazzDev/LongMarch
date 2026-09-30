struct VertexInput {
  [[vk::location(0)]] float2 position : POSITION;
  [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct PixelInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct PanelUniform {
  float4x4 local_to_clip;
};

[[vk::binding(0, 0)]] ConstantBuffer<PanelUniform> panel_uniform : register(b0, space0);
[[vk::binding(0, 1)]] Texture2D<float4> panel_image : register(t0, space1);
[[vk::binding(0, 2)]] SamplerState panel_sampler : register(s0, space2);

PixelInput VSMain(VertexInput input) {
  PixelInput output;
  output.position = mul(panel_uniform.local_to_clip, float4(input.position, 0.0f, 1.0f));
  output.uv = input.uv;
  return output;
}

float4 PSMain(PixelInput input) : SV_TARGET {
  return panel_image.Sample(panel_sampler, input.uv);
}
