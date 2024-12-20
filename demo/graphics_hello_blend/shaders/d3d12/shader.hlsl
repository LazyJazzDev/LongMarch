
struct VSInput {
  float3 position : TEXCOORD0;
  float4 color : TEXCOORD1;
};

struct PSInput {
  float4 position : SV_POSITION;
  float4 color : COLOR;
};

PSInput VSMain(VSInput input) {
  PSInput output;
  output.position = float4(input.position, 1.0f);
  output.color = input.color;
  return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
  return input.color;
}
