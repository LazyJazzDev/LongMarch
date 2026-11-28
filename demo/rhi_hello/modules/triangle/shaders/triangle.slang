struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float3, color);
};

struct Varying {
  float4 position : SV_Position;
  float3 color : COLOR0;
};

Varying VSMain(Input input) {
  Varying output;
  output.position = float4(input.position, 1.0);
  output.color = input.color;
  return output;
}

float4 PSMain(Varying input) : SV_Target0 {
  return float4(input.color, 1.0);
}
