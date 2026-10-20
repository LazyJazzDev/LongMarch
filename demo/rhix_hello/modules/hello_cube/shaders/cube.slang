struct Uniform {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};

RHI_UNIFORM(0, 0, Uniform, globals);

struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float3, color);
};

struct Varying {
  float4 position : SV_Position;
  float3 color : COLOR0;
};

Varying VSMain(Input input) {
  Varying o;
  o.position = mul(globals.projection, mul(globals.view, mul(globals.model, float4(input.position, 1))));
  o.color = input.color;
  return o;
}

float4 PSMain(Varying input) : SV_Target0 {
  return float4(input.color, 1);
}
