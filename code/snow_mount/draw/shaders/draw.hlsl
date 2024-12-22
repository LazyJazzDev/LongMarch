struct VSInput {
  float2 position : TEXCOORD0;
  float2 tex_coord : TEXCOORD1;
  float4 color : TEXCOORD2;
};

struct PSInput {
  float4 position : SV_POSITION;
  float2 tex_coord : TEXCOORD0;
  float4 color : TEXCOORD1;
};

PSInput VSMain(VSInput input) {
  PSInput output;
  output.position = float4(input.position, 0.0, 1.0);
  output.tex_coord = input.tex_coord;
  output.color = input.color;
  return output;
}

struct DrawMetadata {
  float4x4 model;
  float4 color;
};
StructuredBuffer<DrawMetadata> draw_metadata : register(t0, space0);
Texture2D texture : register(t0, space1);
SamplerState sampler : register(s0, space1);

float4 PSMain(PSInput input) : SV_TARGET {
  float4 color = input.color * draw_metadata[0].color *
                 texture.Sample(sampler, input.tex_coord);
}
