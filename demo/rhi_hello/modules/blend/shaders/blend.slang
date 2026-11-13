struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float4, color);
};

struct Varying {
  float4 position : SV_Position;
  RHI_VARYING(0, float4, color);
};

Varying VSMain(Input input) {
  Varying output;
  output.position = float4(input.position, 1.0);
  output.color = input.color;
  return output;
}

float4 PSMain(Varying input) : SV_Target0 {
  return input.color;
}
