
struct UniformBufferObject {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

ConstantBuffer<UniformBufferObject> ubo : register(b0, space0);

struct VSInput {
  [[vk::location(0)]] float3 position : TEXCOORD0;
  [[vk::location(1)]] float3 color : TEXCOORD1;
};

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float4 color : COLOR;
};

PSInput VSMain(VSInput input) {
  PSInput output;
  output.position = float4(input.position, 1.0f);
  output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, output.position)));
  output.color = float4(input.color, 1.0f);
  return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
  return input.color;
}
