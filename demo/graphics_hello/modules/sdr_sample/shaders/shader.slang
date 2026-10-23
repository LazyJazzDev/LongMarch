
float4 VSMain(uint vertex_index : SV_VertexID) : SV_POSITION {
  // Define vertex positions for a rectangle [-0.5, 0.5]^2
  float3 positions[6] = {float3(-0.5f, -0.5f, 0.0f), float3(0.5f, -0.5f, 0.0f), float3(-0.5f, 0.5f, 0.0f),
                         float3(-0.5f, 0.5f, 0.0f),  float3(0.5f, -0.5f, 0.0f), float3(0.5f, 0.5f, 0.0f)};
  return float4(positions[vertex_index], 1.0f);
}

float4 PSMain(float4 pos : SV_POSITION) : SV_TARGET {
  uint2 pixel_coord = uint2(pos.xy);
  float3 color;
  if (pixel_coord.y < 300) {
    color = float3(1.0, 0.0, 0.0);
  } else if (pixel_coord.y < 420) {
    color = float3(0.0, 1.0, 0.0);
  } else {
    color = float3(0.0, 0.0, 1.0);
  }
  if (pixel_coord.x < 640) {
    if (((pixel_coord.x ^ pixel_coord.y) & 1) == 1) {
      return float4(color, 1.0);
    } else {
      return float4(0.0, 0.0, 0.0, 1.0);
    }
  } else {
    float3 linear = color * 0.5;
    // Match an SDR display with a pure gamma 2.2 response.
    float3 encoded = pow(linear, 1.0 / 2.2);
    return float4(encoded, 1.0);
  }
}
