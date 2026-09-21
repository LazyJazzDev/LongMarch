#include "../../compute_contract.hlsli"

struct VSInput {
  [[vk::location(0)]] float3 position : TEXCOORD0;
#if defined(HAS_NORMAL)
  [[vk::location(1)]] float3 normal : TEXCOORD1;

#if defined(HAS_TEXCOORD)
  [[vk::location(2)]] float2 tex_coord : TEXCOORD2;

#if defined(HAS_TANGENT)
  [[vk::location(3)]] float3 tangent : TEXCOORD3;
  [[vk::location(4)]] float signal : TEXCOORD4;
#endif

#else

#if defined(HAS_TANGENT)
  [[vk::location(2)]] float3 tangent : TEXCOORD2;
  [[vk::location(3)]] float signal : TEXCOORD3;
#endif

#endif

#else

#if defined(HAS_TEXCOORD)
  [[vk::location(1)]] float2 tex_coord : TEXCOORD1;

#if defined(HAS_TANGENT)
  [[vk::location(2)]] float3 tangent : TEXCOORD2;
  [[vk::location(3)]] float signal : TEXCOORD3;
#endif

#else

#if defined(HAS_TANGENT)
  [[vk::location(1)]] float3 tangent : TEXCOORD1;
  [[vk::location(2)]] float signal : TEXCOORD2;
#endif

#endif

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

SP_RESOURCE(ConstantBuffer<CameraData>, camera_data, b0, 0);
#define SP_BINDING_camera_data SP_RESOURCE_ACCESS(ConstantBuffer<CameraData>, camera_data, 0)
SP_RESOURCE(ConstantBuffer<GeometryData>, geometry_data, b0, 1);
#define SP_BINDING_geometry_data SP_RESOURCE_ACCESS(ConstantBuffer<GeometryData>, geometry_data, 1)

VSOutput VSMain(VSInput input) {
  VSOutput SP_BINDING_output;
  SP_BINDING_output.world_position = mul(SP_BINDING_geometry_data.model, float4(input.position, 1.0)).xyz;
#if defined(HAS_NORMAL)
  SP_BINDING_output.world_normal = normalize(mul(SP_BINDING_geometry_data.normal_matrix, float4(input.normal, 0)).xyz);
#else
  SP_BINDING_output.world_normal = float3(0, 0, 0);
#endif
#if defined(HAS_TEXCOORD)
  SP_BINDING_output.tex_coord = input.tex_coord;
#else
  SP_BINDING_output.tex_coord = float2(0, 0);
#endif
#if defined(HAS_TANGENT)
  SP_BINDING_output.tangent = normalize(mul(SP_BINDING_geometry_data.model, float4(input.tangent, 0)).xyz);
  SP_BINDING_output.signal = input.signal;
#else
  SP_BINDING_output.tangent = float3(0, 0, 0);
  SP_BINDING_output.signal = 1.0;
#endif
  SP_BINDING_output.position = mul(SP_BINDING_camera_data.view_proj, float4(SP_BINDING_output.world_position, 1.0));
  return SP_BINDING_output;
}
