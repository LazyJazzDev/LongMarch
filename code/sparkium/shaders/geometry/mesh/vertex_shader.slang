struct VSInput {
  RHI_INPUT(0, float3, position);
#if defined(HAS_NORMAL)
  RHI_INPUT(1, float3, normal);
#endif
#if defined(HAS_TEXCOORD)
  RHI_INPUT(2, float2, tex_coord);
#endif
#if defined(HAS_TANGENT)
  RHI_INPUT(3, float3, tangent);
  RHI_INPUT(4, float, signal);
#endif
};

struct VSOutput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float3 world_position : TEXCOORD0;
  [[vk::location(1)]] float3 world_normal : TEXCOORD1;
  [[vk::location(2)]] float2 tex_coord : TEXCOORD2;
  [[vk::location(3)]] float3 tangent : TEXCOORD3;
  [[vk::location(4)]] float signal : TEXCOORD4;
};

struct CameraData {
  float4x4 view;
  float4x4 proj;
  float4x4 view_proj;
  float4x4 inv_view;
  float4x4 inv_proj;
  float4x4 inv_view_proj;
};

struct GeometryData {
  float4x4 model;
  float4x4 inv_model;
  float4x4 normal_matrix;
};

RHI_UNIFORM(0, 0, CameraData, camera_data);
RHI_UNIFORM(0, 1, GeometryData, geometry_data);

VSOutput VSMain(VSInput input) {
  VSOutput output;
  output.world_position = mul(geometry_data.model, float4(input.position, 1.0)).xyz;
#if defined(HAS_NORMAL)
  output.world_normal = normalize(mul(geometry_data.normal_matrix, float4(input.normal, 0)).xyz);
#else
  output.world_normal = float3(0, 0, 0);
#endif
#if defined(HAS_TEXCOORD)
  output.tex_coord = input.tex_coord;
#else
  output.tex_coord = float2(0, 0);
#endif
#if defined(HAS_TANGENT)
  output.tangent = normalize(mul(geometry_data.model, float4(input.tangent, 0)).xyz);
  output.signal = input.signal;
#else
  output.tangent = float3(0, 0, 0);
  output.signal = 1.0;
#endif
  output.position = mul(camera_data.view_proj, float4(output.world_position, 1.0));
  return output;
}
