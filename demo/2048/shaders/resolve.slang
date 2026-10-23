struct ResolveParams {
  float second_frame_alpha;
  uint scale;
  uint2 padding;
};

[[vk::binding(0, 0)]] ConstantBuffer<ResolveParams> params : register(b0, space0);
[[vk::binding(0, 1)]] Texture2D<float4> main_frame : register(t0, space1);
[[vk::binding(0, 2)]] Texture2D<float4> second_frame : register(t0, space2);

float4 VSMain(uint vertex_index : SV_VertexID) : SV_POSITION {
  // A single triangle covering the whole viewport.
  float2 uv = float2((vertex_index << 1) & 2, vertex_index & 2);
  return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Averages the supersampled pixels covered by one output pixel, then blends
// the optional second frame over the main frame.
float4 PSMain(float4 position : SV_POSITION) : SV_TARGET {
  int2 base = int2(position.xy) * int(params.scale);
  float4 main_color = float4(0.0, 0.0, 0.0, 0.0);
  float4 second_color = float4(0.0, 0.0, 0.0, 0.0);
  for (uint y = 0; y < params.scale; y++) {
    for (uint x = 0; x < params.scale; x++) {
      int3 coord = int3(base + int2(x, y), 0);
      main_color += main_frame.Load(coord);
      second_color += second_frame.Load(coord);
    }
  }
  float inv_samples = 1.0 / float(params.scale * params.scale);
  main_color *= inv_samples;
  second_color *= inv_samples;
  float3 color = lerp(main_color.rgb, second_color.rgb, params.second_frame_alpha);
  return float4(color, 1.0);
}
