
struct PSInput {
  float4 position : SV_POSITION;
  float2 tex_coord : TEXCOORD;
};

Texture2D<float4> src_texture : register(t0);
SamplerState src_sampler : register(s0);

PSInput VSMain(uint vertex_id : SV_VertexID) {
  float2 vertices[6] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
                        float2(-1.0, -1.0), float2(1.0, 1.0),  float2(-1.0, 1.0)};

  PSInput result;
  result.position = float4(vertices[vertex_id], 0.0, 1.0);
  result.tex_coord = 0.5 * (vertices[vertex_id] + 1.0);
  result.tex_coord.y = 1.0 - result.tex_coord.y;

  return result;
}

float4 PSMain(PSInput input) : SV_TARGET {
  return src_texture.Sample(src_sampler, input.tex_coord);
}
