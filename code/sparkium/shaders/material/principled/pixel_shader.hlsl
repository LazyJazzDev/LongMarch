#include "buffer_helper.hlsli"

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float3 world_position : TEXCOORD0;
  [[vk::location(1)]] float3 world_normal : TEXCOORD1;
  [[vk::location(2)]] float2 tex_coord : TEXCOORD2;
  [[vk::location(3)]] float3 tangent : TEXCOORD3;
  [[vk::location(4)]] float signal : TEXCOORD4;
};

struct PSOutput {
  float4 radiance : SV_TARGET0;
  float4 albedo_roughness : SV_TARGET1;
  float4 position_specular : SV_TARGET2;
  float4 normal_metallic : SV_TARGET3;
  int stencil : SV_TARGET4;
};

ByteAddressBuffer material_data : register(t0, space2);
Texture2D<float4> textures[] : register(t0, space3);
SamplerState S : register(s0, space4);

PSOutput PSMain(PSInput input) {
  PSOutput output;
  float3 geom_normal;
  // compute geometry normal from position derivatives
  float3 dp1 = ddx(input.world_position);
  float3 dp2 = ddy(input.world_position);
  geom_normal = normalize(cross(dp2, dp1));
  if (length(input.world_normal) < 0.001) {
    input.world_normal = geom_normal;
  }
  float3 N = normalize(input.world_normal);
  float3 T = input.tangent;
  float3 B = float3(0, 0, 0);
  if (length(T) > 0.001) {
    T = normalize(T - dot(T, N) * N);
    B = cross(N, T) * input.signal;
  }

  StreamedBufferReference<ByteAddressBuffer> material_buffer = MakeStreamedBufferReference(material_data, 0);

  float3 base_color = material_buffer.LoadFloat3();
  float3 subsurface_color = material_buffer.LoadFloat3();
  float subsurface = material_buffer.LoadFloat();
  float3 subsurface_radius = material_buffer.LoadFloat3();
  float metallic = material_buffer.LoadFloat();
  float specular = material_buffer.LoadFloat();
  float specular_tint = material_buffer.LoadFloat();
  float roughness = material_buffer.LoadFloat();
  float anisotropic = material_buffer.LoadFloat();
  float anisotropic_rotation = material_buffer.LoadFloat();
  float sheen = material_buffer.LoadFloat();
  float sheen_tint = material_buffer.LoadFloat();
  float clearcoat = material_buffer.LoadFloat();
  float clearcoat_roughness = material_buffer.LoadFloat();
  float ior = material_buffer.LoadFloat();
  float transmission = material_buffer.LoadFloat();
  float transmission_roughness = material_buffer.LoadFloat();

  float3 emission = material_buffer.LoadFloat3();
  float strength = material_buffer.LoadFloat();

  float y_signal = material_buffer.LoadFloat();

  input.tex_coord.y = 1.0 - input.tex_coord.y;

  int use_texture;
  use_texture = material_buffer.LoadInt();
  if (use_texture)
    base_color = textures[0].Sample(S, input.tex_coord).xyz;
  use_texture = material_buffer.LoadInt();
  if (use_texture)
    roughness = textures[1].Sample(S, input.tex_coord).x;
  use_texture = material_buffer.LoadInt();
  if (use_texture)
    specular = textures[2].Sample(S, input.tex_coord).x;
  use_texture = material_buffer.LoadInt();
  if (use_texture)
    metallic = textures[3].Sample(S, input.tex_coord).x;
  use_texture = material_buffer.LoadInt();
  if (use_texture) {
    float3 tbn = textures[4].Sample(S, input.tex_coord).xyz;

    if (length(tbn) > 0.001 && length(B) > 0.001) {
      tbn = tbn * 2.0f - 1.0f;
      tbn = normalize(tbn);
      N = normalize(mul(tbn, float3x3(T, B * y_signal, N)));
    }
  }

  output.radiance = float4(emission * strength, 0.0);
  // base_color = float3(max(input.signal, 0.0), 0.0, max(-input.signal, 0.0));
  // base_color = T * 0.5 + 0.5;
  // base_color = B * 0.5 + 0.5;
  // base_color = float3(input.tex_coord, 0.0);
  output.albedo_roughness = float4(base_color, roughness);
  output.position_specular = float4(input.world_position, specular);
  output.normal_metallic = float4(N * 0.5 + 0.5, metallic);
  output.stencil = 0;
  return output;
}
