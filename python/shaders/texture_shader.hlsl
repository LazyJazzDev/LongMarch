Texture2D texture : register(t0, space0);
SamplerState samp : register(s0, space1);

struct VSInput {
  [[vk::location(0)]] float3 position : TEXCOORD0;
  [[vk::location(1)]] float2 tex_coord : TEXCOORD1;
};

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 tex_coord : TEXCOORD0;
};

PSInput VSMain(VSInput input) {
  PSInput output;
  output.position = float4(input.position, 1.0f);
  output.tex_coord = input.tex_coord;
  return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
  return texture.Sample(samp, input.tex_coord);
}
