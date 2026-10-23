struct InstanceInfo {
  float4x4 model;
  float4 color;
  uint4 extra;
};

struct GlobalUniformObject {
  float4x4 view;
};

[[vk::binding(0, 0)]] ByteAddressBuffer instance_infos : register(t0, space0);
[[vk::binding(0, 1)]] ConstantBuffer<GlobalUniformObject> global_uniform : register(b0, space1);

struct VSInput {
  [[vk::location(0)]] float2 position : TEXCOORD0;
  [[vk::location(1)]] float4 color : TEXCOORD1;
  [[vk::location(2)]] uint instance_id : TEXCOORD2;
};

struct PSInput {
  float4 position : SV_POSITION;
  [[vk::location(0)]] float2 local_position : TEXCOORD0;
  [[vk::location(1)]] float4 color : TEXCOORD1;
  [[vk::location(2)]] nointerpolation uint4 extra : TEXCOORD2;
};

PSInput VSMain(VSInput input) {
  InstanceInfo instance_info = instance_infos.Load<InstanceInfo>(sizeof(InstanceInfo) * input.instance_id);
  float4 screen_pos = mul(instance_info.model, float4(input.position, 0.0, 1.0));
  PSInput output;
  output.position = mul(global_uniform.view, screen_pos);
  output.color = input.color * instance_info.color;
  output.local_position = input.position;
  if (instance_info.extra.x == 3) {
    output.local_position = screen_pos.xy;
  }
  output.extra = instance_info.extra;
  return output;
}

float4 IconTheme(PSInput input) {
  float scale = 2.0 - length(input.local_position - float2(-1.0, -1.0)) * 0.3;
  return float4(input.color.rgb * scale, 1.0);
}

float4 CellTheme(PSInput input) {
  float scale = 1.0 + length(input.local_position - float2(-1.0, -1.0)) * 0.15;
  float4 background_color = float4(input.color.rgb * scale * asfloat(input.extra.y), 1.0);

  scale = 2.0 - length(input.local_position - float2(-1.0, -1.0)) * 0.3;
  float4 foreground_color = float4(input.color.rgb * scale * asfloat(input.extra.z), 1.0) * asfloat(input.extra.w);
  return float4(background_color.rgb * (1.0 - foreground_color.a) + foreground_color.rgb * foreground_color.a, 1.0);
}

float4 PSMain(PSInput input) : SV_TARGET {
  float4 color;
  switch (input.extra.x) {
    case 0:
      color = input.color;
      break;
    case 1:
      color = IconTheme(input);
      break;
    case 2:
      color = CellTheme(input);
      break;
    default:
      color = float4(1.0, 1.0, 1.0, 1.0);
      break;
  }
  return color;
}
