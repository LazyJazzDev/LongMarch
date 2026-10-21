RHI_TEXTURE2D(0, 0, float4, input_image);

struct Vertex {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

[shader("vertex")] Vertex VSMain(uint id
                                 : SV_VertexID) {
  Vertex v;
  v.uv = float2((id << 1) & 2, id & 2);
  v.position = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return v;
}[shader("fragment")] float4 PSMain(Vertex v)
    : SV_Target0 {
  uint w, h;
  input_image.GetDimensions(w, h);
  return input_image.Load(int3(min(uint2(v.uv * float2(w, h)), uint2(w - 1, h - 1)), 0));
}
