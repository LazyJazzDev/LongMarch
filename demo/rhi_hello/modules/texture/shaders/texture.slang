RHI_TEXTURE2D(0, 0, float4, albedo);
RHI_SAMPLER(0, 1, linearSampler);

struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float2, uv);
};

struct Varying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

Varying VSMain(Input input) {
  Varying o;
  o.position = float4(input.position, 1);
  o.uv = input.uv;
  return o;
}

float4 PSMain(Varying input) : SV_Target0 {
  return albedo.Sample(linearSampler, input.uv);
}
