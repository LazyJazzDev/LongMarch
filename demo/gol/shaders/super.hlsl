struct InstanceInfo {
  float4x4 model;
  float4 color;
  uint4 extra;
  float4 clip_rect;
};

struct GlobalUniformObject {
  float4x4 view;
};

// Distance of a die plate from the die centre, matching FaceFrame() in randomize_button.cpp,
// and the virtual camera distance of the die's slight perspective divide.
static const float kDicePlateOffset = 0.43;
static const float kDicePerspectiveDistance = 0.6;

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
  [[vk::location(3)]] float2 screen_position : TEXCOORD3;
  [[vk::location(4)]] nointerpolation float4 clip_rect : TEXCOORD4;
  [[vk::location(5)]] noperspective float2 projected_die_position : TEXCOORD5;
};

PSInput VSMain(VSInput input) {
  InstanceInfo instance_info = instance_infos.Load<InstanceInfo>(sizeof(InstanceInfo) * input.instance_id);
  float4 screen_pos = mul(instance_info.model, float4(input.position, 0.0, 1.0));
  PSInput output;
  output.position = mul(global_uniform.view, screen_pos);
  output.projected_die_position = 0.0;
  if (instance_info.extra.x == 3) {
    // The randomize die converges slightly toward a vanishing point. Every plate is offset
    // from the shared die centre, so recovering that centre keeps the seams aligned while
    // the die tumbles. Other instances keep the orthographic placement.
    float4 plate_centre = mul(instance_info.model, float4(0.0, 0.0, 0.0, 1.0));
    float4 die_centre = plate_centre - kDicePlateOffset * mul(instance_info.model, float4(0.0, 0.0, 1.0, 0.0));
    float w = 1.0 + (screen_pos.z - die_centre.z) / kDicePerspectiveDistance;
    float4 centre_clip = mul(global_uniform.view, die_centre);
    output.position.xy += (w - 1.0) * centre_clip.xy;
    output.position.z *= w;
    output.position.w = w;
    // Divide by the same perspective w as the clip position, then interpolate
    // without perspective correction so the lighting stays fixed on screen.
    output.projected_die_position = (screen_pos.xy - die_centre.xy) / (w * asfloat(instance_info.extra.zw));
  }
  output.color = input.color * instance_info.color;
  output.local_position = input.position;
  output.extra = instance_info.extra;
  output.screen_position = screen_pos.xy;
  output.clip_rect = instance_info.clip_rect;
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

// A rounded plate with real cutouts. Discard exposes the button background through
// the holes and seams; the existing supersampled resolve antialiases their edges.
float4 DiceFaceTheme(PSInput input) {
  float2 p = input.local_position;
  float2 q = abs(p) - 0.74;
  float rounded_rect = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 0.26;
  clip(-rounded_rect);
  uint value = input.extra.y;
  float hole = 10.0;
  if ((value & 1) != 0)
    hole = length(p);
  if (value >= 2)
    hole = min(hole, min(length(p - float2(-0.50, -0.50)), length(p - float2(0.50, 0.50))));
  if (value >= 4)
    hole = min(hole, min(length(p - float2(-0.50, 0.50)), length(p - float2(0.50, -0.50))));
  if (value == 6)
    hole = min(hole, min(length(p - float2(-0.50, 0.0)), length(p - float2(0.50, 0.0))));
  clip(hole - 0.20);
  float2 projected = input.projected_die_position;
  float shade = 1.0 - 0.25 * (projected.x + projected.y);
  return float4(input.color.rgb * shade, 1.0);
}

float4 SliderTheme(PSInput input) {
  float2 half_size = float2(asfloat(input.extra.y), asfloat(input.extra.z));
  float radius = asfloat(input.extra.w);
  float2 q = abs(input.local_position) * half_size - half_size + radius;
  float distance = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
  clip(-distance);
  // Shape the cross-section in framebuffer pixels, not stretched square UVs.
  // The short side sets the rounding width on every edge, including the ends.
  float2 position = input.local_position * half_size;
  float half_thickness = max(min(half_size.x, half_size.y), 0.001);
  float2 edge_distance = half_size - abs(position);
  float2 slope = sign(position) * (1.0 - smoothstep(0.0, half_thickness, edge_distance));
  float relief = input.extra.x == 5 ? -1.0 : 1.0;
  // A shallow, monotonic light ramp avoids the off-center highlight peaks of
  // normalized curved normals. Keep opposite shading for raised/recessed areas.
  float shade = 1.0 - relief * 0.065 * (slope.x + slope.y);
  return float4(input.color.rgb * shade, 1.0);
}

uint RandPCG(inout uint rng_state) {
  uint state = rng_state;
  rng_state = rng_state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  rng_state = (word >> 22u) ^ word;
  return rng_state;
}

float RandFloat(inout uint rng_state) {
  return float(rng_state) / 4294967296.0;
}

float ShakeValue(inout uint rng_state, float value, float shake) {
  return value + (RandFloat(rng_state) - 0.5) * shake;
}

float4 ShakeColor(inout uint rng_state, float4 color, float shake) {
  float r = ShakeValue(rng_state, color.r, shake);
  float g = ShakeValue(rng_state, color.g, shake);
  float b = ShakeValue(rng_state, color.b, shake);
  return clamp(float4(r, g, b, color.a), 0.0, 1.0);
}

float4 PSMain(PSInput input) : SV_TARGET {
  clip(input.screen_position - input.clip_rect.xy);
  clip(input.clip_rect.zw - input.screen_position);
  // Init rng_state with the fragment position, one dimension at a time.
  uint rng_state = uint(input.position.x);
  rng_state = RandPCG(rng_state);
  rng_state ^= uint(input.position.y);
  rng_state = RandPCG(rng_state);

  float4 color;
  switch (input.extra.x) {
    case 0:
      color = input.color;
      break;
    case 1:
      color = IconTheme(input);
      break;
    case 3:
      color = DiceFaceTheme(input);
      break;
    case 4:
    case 5:
      color = SliderTheme(input);
      break;
    case 2:
      color = CellTheme(input);
      break;
    default:
      color = float4(1.0, 1.0, 1.0, 1.0);
      break;
  }

  // Dither to hide banding in the gradients.
  return ShakeColor(rng_state, color, 1.0 / 255.0);
}
