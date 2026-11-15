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

RHI_BYTE_ADDRESS_BUFFER(0, 0, instance_infos);
RHI_BYTE_ADDRESS_BUFFER(0, 2, cell_bits);
RHI_BYTE_ADDRESS_BUFFER(0, 3, cell_mipmaps);
RHI_UNIFORM(0, 1, GlobalUniformObject, global_uniform);

struct VSInput {
  RHI_INPUT(0, float2, position);
  RHI_INPUT(1, float4, color);
  RHI_INPUT(2, uint, instance_id);
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
  // Keep a single uint view of the byte buffer. Loading the mixed-type struct
  // directly emits aliased float/uint descriptors and crashes the Maleoon 910
  // pipeline compiler. Instance matrices are stored as four column vectors.
  uint offset = sizeof(InstanceInfo) * input.instance_id;
  InstanceInfo instance_info;
  instance_info.model =
      transpose(float4x4(asfloat(instance_infos.Load4(offset)), asfloat(instance_infos.Load4(offset + 16)),
                         asfloat(instance_infos.Load4(offset + 32)), asfloat(instance_infos.Load4(offset + 48))));
  instance_info.color = asfloat(instance_infos.Load4(offset + 64));
  instance_info.extra = instance_infos.Load4(offset + 80);
  instance_info.clip_rect = asfloat(instance_infos.Load4(offset + 96));
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

// The mobile world uses 64x64 compute tiles. Each tile stores exact counts
// for levels 1..6 in 219 words; coarser levels are global uint count arrays.
uint GridMipCount(uint level, int2 coordinate) {
  uint side = 16384 >> level;
  uint2 p = uint2(clamp(coordinate, int2(0), int2(side - 1)));
  if (level == 0) {
    uint page = (p.y / 256) * 64 + p.x / 256;
    return (cell_bits.Load((page * 2048 + (p.y & 255) * 8 + (p.x & 255) / 32) * 4) >> (p.x & 31)) & 1;
  }
  if (level <= 6) {
    uint tile_shift = 6 - level;
    uint local_mask = (1u << tile_shift) - 1;
    uint tile = ((p.y >> tile_shift) << 8) + (p.x >> tile_shift);
    uint offset = level == 1 ? 0 : level == 2 ? 128 : level == 3 ? 192 : level == 4 ? 208 : level == 5 ? 216 : 218;
    uint bit_shift = level == 1 ? 2 : level <= 3 ? 3 : 4;
    uint bits = 1u << bit_shift;
    uint word_shift = 5 - bit_shift;
    uint index = ((p.y & local_mask) << tile_shift) | (p.x & local_mask);
    uint packed = cell_mipmaps.Load((tile * 219 + offset + (index >> word_shift)) * 4);
    return (packed >> ((index & ((1u << word_shift) - 1)) << bit_shift)) & (0xffffffffu >> (32 - bits));
  }
  uint offset = 65536 * 219 + (65536 - (1u << (30 - 2 * level))) / 3;
  return cell_mipmaps.Load((offset + (p.y << (14 - level)) + p.x) * 4);
}

float GridMipSample(uint level, float2 grid) {
  float2 p = grid / float(1u << level) - 0.5;
  int2 origin = int2(floor(p));
  float2 f = frac(p);
  float a = lerp(float(GridMipCount(level, origin)), float(GridMipCount(level, origin + int2(1, 0))), f.x);
  float b = lerp(float(GridMipCount(level, origin + int2(0, 1))), float(GridMipCount(level, origin + 1)), f.x);
  return lerp(a, b, f.y) / float(1u << (level * 2));
}

float GridFade(PSInput input) {
  float2 edge = min(input.screen_position - input.clip_rect.xy, input.clip_rect.zw - input.screen_position);
  float width = max(1.0, min(input.clip_rect.z - input.clip_rect.x, input.clip_rect.w - input.clip_rect.y) * 0.06);
  float opacity = float(input.extra.y >> 18) / 16383.0;
  return lerp(1.0, smoothstep(0.0, width, min(edge.x, edge.y)), opacity);
}

uint DetailSpan(uint level) {
  return 2048u >> (3 * level);
}

float3 CheckerColor(float3 color, float2 grid, uint packed) {
  uint level = (packed >> 16) & 3u;
  float t = float(packed & 65535u) / 65535.0;
  if (level == 3)
    return color;
  float parent_c = 0.02 * (1.0 - t), child_c = level == 2 ? 0.0 : 0.02 * t;
  uint2 parent_cell = uint2(floor(grid / float(DetailSpan(level))));
  float parent_a = ((parent_cell.x + parent_cell.y) & 1u) == 0 ? 1.0 : 0.0;
  if (t > 0 && level < 3) {
    uint2 child_cell = uint2(floor(grid / float(DetailSpan(level + 1))));
    float child_a = ((child_cell.x + child_cell.y) & 1u) == 0 ? 1.0 : 0.0;
    color = color * (1.0 - child_c) + child_a * child_c;
  }
  return color * (1.0 - parent_c) + parent_a * parent_c;
}

// One quad covers the visible grid: color.xy is the top-left corner of cell (0, 0),
// color.z the cell pitch in framebuffer pixels, color.w the brightness of the gaps,
// extra.zw the grid dimensions and extra.y the byte offset of one packed
// appearance word per cell.
float4 CellGridTheme(PSInput input, float sample_size, bool allow_discard = true) {
  float2 grid = (input.screen_position - input.color.xy) / input.color.z;
  int2 cell = int2(floor(grid));
  int2 size = int2(input.extra.zw);
  clip(float2(cell));
  clip(float2(size - 1 - cell));
  float3 filtered = 0.0;
  if (input.extra.x == 6 && input.color.z < 8.0) {
    float lod = clamp(log2(1.0 / input.color.z), 0.0, 14.0);
    uint level = uint(floor(lod));
    float density = GridMipSample(level, grid);
    if (lod > 0)
      density = lerp(density, GridMipSample(min(level + 1, 14u), grid), frac(lod));
    // Area averages of the same rounded, shaded dead/live cell themes below,
    // including the inter-cell gaps. Blend levels in linear brightness.
    float brightness = lerp(0.183960325, 0.493834190, density);
    filtered = float3(brightness);
    if (input.color.z <= 2.0)
      return float4(lerp(float3(input.color.w), CheckerColor(filtered, grid, input.extra.y), GridFade(input)), 1.0);
  }
  // Each cell fills the middle 80% of its pitch, mapped to [-1, 1] as before.
  float2 local_position = (grid - float2(cell) - 0.5) / 0.4;
  // The original 120-triangle silhouette samples |x|^8 + |y|^8 = 1. The L8
  // norm minus one is a distance to that edge in cell units (exact along the
  // axes); in rendered samples it antialiases cells a few pixels wide without
  // supersampling.
  float2 p = local_position * local_position;
  p *= p;
  p *= p;
  float samples_per_unit = 0.4 * input.color.z / max(sample_size, 1.0e-6);
  float coverage = saturate(0.5 - (pow(p.x + p.y, 0.125) - 1.0) * samples_per_unit);
  if (allow_discard && input.extra.x != 6)
    clip(coverage - 1.0 / 512.0);
  float3 appearance;
  if (input.extra.x == 6) {
    uint tile = (uint(cell.y) / 256) * 64 + uint(cell.x) / 256;
    uint word = tile * 2048 + (uint(cell.y) & 255) * 8 + (uint(cell.x) & 255) / 32;
    uint alive = (cell_bits.Load(word * 4) >> (uint(cell.x) & 31)) & 1;
    appearance = float3(0.2, 0.48, float(alive));
  } else {
    uint packed = instance_infos.Load(input.extra.y + 4 * uint(cell.y * size.x + cell.x));
    appearance = float3(packed & 1023u, (packed >> 10) & 1023u, (packed >> 20) & 1023u) / 1023.0;
  }

  float scale = 1.0 + length(local_position - float2(-1.0, -1.0)) * 0.15;
  float3 background_color = scale * appearance.x;
  scale = 2.0 - length(local_position - float2(-1.0, -1.0)) * 0.3;
  float4 foreground_color = float4(float3(scale * appearance.y), 1.0) * appearance.z;
  float3 cell_color = background_color * (1.0 - foreground_color.a) + foreground_color.rgb * foreground_color.a;
  float fade = input.extra.x == 6 ? GridFade(input) : 1.0;
  float3 shaded = lerp(float3(input.color.w), cell_color, coverage);
  if (input.extra.x == 6) {
    // Rounded bevels and narrow gaps are high-frequency detail. Remove them
    // continuously before they become subpixel; keep population filtering active
    // through the transition rather than switching abruptly at one cell/pixel.
    if (input.color.z < 8.0)
      shaded = lerp(filtered, shaded, smoothstep(2.0, 8.0, input.color.z));
    shaded = CheckerColor(shaded, grid, input.extra.y);
  }
  return float4(lerp(float3(input.color.w), shaded, fade), 1.0);
}

float RulerLayer(PSInput input, uint span, float font, float prominence, bool major) {
  float2 p = input.screen_position;
  float2 origin = input.color.xy;
  float spacing = float(span) * input.color.z;
  float tick_length = font * lerp(0.45, 0.65, prominence);
  float width = max(1.0, font / 14.0);
  float pad = max(2.0, font * 0.15);
  float2 edge = min(p - input.clip_rect.xy, input.clip_rect.zw - p);
  float2 rounded = round((p - origin) / spacing);
  float2 tick = abs(p - (origin + rounded * spacing));
  float result = 0;
  if (rounded.x >= 0 && rounded.x * span <= 16384 && edge.y < tick_length)
    result = max(result, saturate(width - tick.x));
  if (rounded.y >= 0 && rounded.y * span <= 16384 && edge.x < tick_length)
    result = max(result, saturate(width - tick.y));
  return result;
}

float4 GridRulerTheme(PSInput input, float sample_size) {
  uint level = (input.extra.y >> 16) & 3u;
  float t = float(input.extra.y & 65535u) / 65535.0;
  float font = float(input.extra.z) / 64.0;
  // At t=1 the child has exactly the next major layer's dimensions. Shared
  // multiples retain the parent label, avoiding two overlapping glyphs; their
  // size and position already match the child when it becomes the major layer.
  float mask = RulerLayer(input, DetailSpan(level), font, 1.0, true);
  if (t > 0 && level < 3)
    mask = max(mask, t * RulerLayer(input, DetailSpan(level + 1), font * lerp(0.5, 1.0, t), t, false));
  float2 edge = min(input.screen_position - input.clip_rect.xy, input.clip_rect.zw - input.screen_position);
  mask = max(mask, 0.3 * saturate(max(1.0, font / 18.0) - min(edge.x, edge.y)));
  if (input.extra.x == 8)
    mask = 1.0;
  clip(mask - 0.001);
  float3 background = float3(0.08);
  float2 grid = (input.screen_position - input.color.xy) / input.color.z;
  if (all(grid >= 0) && all(grid < 16384)) {
    PSInput under = input;
    under.extra = uint4(6, input.extra.y, 16384, 16384);
    under.color.w = 0.08;
    background = CellGridTheme(under, sample_size, false).rgb;
  }
  return float4(lerp(background, float3(0.85), mask * input.color.w), 1);
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
  float shade = 1.0 - 0.16 * (projected.x + projected.y);
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
  // Framebuffer pixels per rendered sample (below 1 when supersampled), taken
  // before any clip from a linear input so it is valid across the pixel quad.
  float sample_size = max(abs(ddx(input.screen_position.x)), abs(ddy(input.screen_position.y)));
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
    case 8:
    case 7:
      color = GridRulerTheme(input, sample_size);
      break;
    case 2:
    case 6:
      color = CellGridTheme(input, sample_size);
      break;
    default:
      color = float4(1.0, 1.0, 1.0, 1.0);
      break;
  }

  // Presentation is opaque, including the direct single-frame path.
  color.a = 1.0;
  // Dither to hide banding in the gradients.
  return ShakeColor(rng_state, color, 1.0 / 255.0);
}
