#include "inverse.hlsli"

struct VSInput {
  [[vk::location(0)]] float3 position : TEXCOORD0;
  [[vk::location(1)]] float3 normal : TEXCOORD1;
  [[vk::location(2)]] float2 texcoord : TEXCOORD2;
  [[vk::location(3)]] float4 color : TEXCOORD3;
};

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float3 world_position : TEXCOORD0;
  [[vk::location(1)]] float3 normal : TEXCOORD1;
  [[vk::location(2)]] float2 texcoord : TEXCOORD2;
  [[vk::location(3)]] float4 color : TEXCOORD3;
};

typedef PSInput GSInput;

struct PSOutput {
  float4 exposure : SV_TARGET0;
  float4 albedo : SV_TARGET1;
  float4 position : SV_TARGET2;
  float4 normal : SV_TARGET3;
};

struct CameraInfo {
  float4x4 proj;
  float4x4 view;
};

struct Material {
  float4 albedo;
};

struct EntityInfo {
  Material material;
  float4x4 model;
};

ConstantBuffer<CameraInfo> camera_info : register(b0, space0);
ConstantBuffer<EntityInfo> entity_info : register(b0, space1);

GSInput VSMain(VSInput input) {
  GSInput output = (PSInput)0;
  output.position = mul(entity_info.model, float4(input.position, 1.0f));
  output.world_position = output.position.xyz;
  output.position = mul(camera_info.view, output.position);
  output.position = mul(camera_info.proj, output.position);
  output.normal =
      mul(transpose(inverse(float3x3(entity_info.model[0].xyz, entity_info.model[1].xyz, entity_info.model[2].xyz))),
          input.normal);
  output.texcoord = input.texcoord;
  output.color = input.color * entity_info.material.albedo;
  return output;
}

[maxvertexcount(3)] void GSMain(triangle GSInput input[3], inout TriangleStream<PSInput> triStream) {
  float3 v0 = input[1].world_position - input[0].world_position;
  float3 v1 = input[2].world_position - input[0].world_position;
  float3 normal = normalize(cross(v0, v1));
  PSInput output = (PSInput)0;
  [unroll] for (int i = 0; i < 3; i++) {
    output.position = input[i].position;
    output.world_position = input[i].world_position;
    output.normal = normal;
    output.texcoord = input[i].texcoord;
    output.color = input[i].color;
    triStream.Append(output);
  }
  triStream.RestartStrip();
}

PSOutput PSMain(PSInput input, bool is_front_facing
                : SV_IsFrontFace) {
  float3 normal = normalize(input.normal);
  float4 filter_color = float4(1.0, 0.5, 0.5, 1.0);
  if (!is_front_facing) {
    normal = -normal;
    filter_color = float4(0.5, 0.5, 1.0, 1.0);
  }
  PSOutput output = (PSOutput)0;
  // out_exposure = vec4(frag_color.rgb * (max(0.0, dot(normal,
  // normalize(vec3(3.0, 1.0, 2.0)))) * 0.5 + 0.5), 1.0)
  output.exposure = float4(0., 0., 0., 1.0f);
  // float4(input.color.rgb * (max(0.0, dot(normal, normalize(float3(3.0, 1.0, 2.0)))) * 0.5), 1.0f);
  output.albedo = input.color;
  output.position = float4(input.world_position, 1.0f);
  output.normal = float4(normal, 1.0f);
  return output;
}
