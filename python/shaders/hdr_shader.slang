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
  output.color = float4(input.color, 1.0f);
  return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
  return float4(pow(input.color.rgb, 2.2), 1.0f);
}
