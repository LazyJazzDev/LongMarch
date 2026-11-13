struct Uniform {
  float4x4 transform;
};

RHI_UNIFORM(0, 0, Uniform, globals);

struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float3, normal);
  RHI_INPUT(2, float3, color);
};

struct Varying {
  float4 position : SV_Position;
  float3 normal : NORMAL;
  float3 color : COLOR0;
};

Varying VSMain(Input input) {
  Varying output;
  output.position = mul(globals.transform, float4(input.position, 1));
  output.normal = input.normal;
  output.color = input.color;
  return output;
}

float4 PSMain(Varying input) : SV_Target0 {
  float light =
      length(input.normal) < 0.1 ? 1 : 0.28 + 0.72 * abs(dot(normalize(input.normal), normalize(float3(-0.5, 1, 0.3))));
  return float4(input.color * light, 1);
}
