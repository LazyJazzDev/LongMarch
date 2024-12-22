struct VSInput {
  float2 position : TEXCOORD0;
  float2 tex_coord : TEXCOORD1;
  float4 color : TEXCOORD2;
  uint instance_id : TEXCOORD3;
};

struct PSInput {
  float4 position : SV_POSITION;
  float2 tex_coord : TEXCOORD0;
  float4 color : TEXCOORD1;
};

struct DrawMetadata {
  float4x4 model;
  float4 color;
};

StructuredBuffer<DrawMetadata> draw_metadata : register(t0, space0);
Texture2D texture : register(t0, space1);
SamplerState samp : register(s0, space2);

PSInput VSMain(VSInput input) {
  DrawMetadata metadata =
      draw_metadata[input.instance_id];  //.Load<DrawMetadata>(0 * 80);
  PSInput output;
  output.position = mul(metadata.model, float4(input.position, 0.0, 1.0));
  output.tex_coord = input.tex_coord;
  output.color = input.color * metadata.color;
  return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
  float4 color = input.color * texture.Sample(samp, input.tex_coord);
  return color;
}
