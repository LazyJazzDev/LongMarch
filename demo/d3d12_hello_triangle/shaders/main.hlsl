
struct VSInput {
  float3 position : ATTRIBUTE0;
  float3 color : ATTRIBUTE1;
};

struct PSInput {
  float4 position : SV_POSITION;
  float3 color : COLOR;
};

[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")] PSInput VSMain(
    VSInput input) {
  PSInput output;
  output.position = float4(input.position, 1.0f);
  output.color = input.color;
  return output;
}

float4 PSMain(PSInput input)
    : SV_TARGET {
  return float4(input.color, 1.0f);
}
