struct Uniform {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
};

RHI_UNIFORM(0, 0, Uniform, globals);
RHI_TEXTURE2D(0, 1, float4, image);
RHI_SAMPLER(0, 2, filtering);

struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float2, uv);
};

struct Varying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

Varying VSMain(Input input) {
  Varying result;
  result.position = mul(globals.projection, mul(globals.view, mul(globals.model, float4(input.position, 1))));
  result.uv = input.uv;
  return result;
}

float4 PSMain(Varying input) : SV_Target0 {
  return image.Sample(filtering, input.uv);
}
