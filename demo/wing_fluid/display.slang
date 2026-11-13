struct Settings {
  uint width;
  uint height;
  uint mode;
  float speed;
  float dye_strength;
  float angle;
  uint smooth_boundary;
  float padding;
};

RHI_UNIFORM(0, 0, Settings, settings);
RHI_BYTE_ADDRESS_BUFFER(0, 1, field);

struct Varying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

Varying VSMain(uint id : SV_VertexID) {
  Varying o;
  o.uv = float2((id << 1) & 2, id & 2);
  o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return o;
}

float4 Cell(int2 p) {
  p = clamp(p, int2(0), int2(settings.width - 1, settings.height - 1));
  return field.Load<float4>((p.y * settings.width + p.x) * 16);
}

float3 Ramp(float t) {
  float3 a = float3(0.012, 0.025, 0.075), b = float3(0.01, 0.28, 0.38);
  float3 c = float3(0.2, 0.72, 0.53), d = float3(1.0, 0.38, 0.06);
  t = saturate(t) * 3;
  return t < 1 ? lerp(a, b, t) : t < 2 ? lerp(b, c, t - 1) : lerp(c, d, t - 2);
}

float4 PSMain(Varying input) : SV_Target0 {
  float2 pos = float2(input.uv.x, 1 - input.uv.y) * float2(settings.width, settings.height) - 0.5;
  int2 p = int2(floor(pos));
  float2 f = frac(pos);
  float4 nearest = Cell(int2(round(pos)));
  if (!settings.smooth_boundary && nearest.w < 0)
    return float4(0.82, 0.87, 0.88, 1);
  float4 q = lerp(lerp(Cell(p), Cell(p + int2(1, 0)), f.x), lerp(Cell(p + int2(0, 1)), Cell(p + int2(1, 1)), f.x), f.y);
  float speed = max(settings.speed, 0.1);
  float3 color;
  if (settings.mode == 2) {
    float curl =
        ((Cell(p + int2(1, 0)).y - Cell(p - int2(1, 0)).y) - (Cell(p + int2(0, 1)).x - Cell(p - int2(0, 1)).x)) *
        settings.width / 12.0;
    float v = tanh(curl / (speed * 8));
    color = lerp(float3(0.025, 0.035, 0.065), v > 0 ? float3(0.95, 0.18, 0.055) : float3(0.025, 0.48, 0.9), abs(v));
  } else if (settings.mode == 3) {
    float v = tanh(q.z / (speed * speed));
    color = lerp(float3(0.06, 0.07, 0.095), v > 0 ? float3(0.95, 0.24, 0.075) : float3(0.025, 0.35, 0.95), abs(v));
  } else {
    color = Ramp(length(q.xy) / (speed * 2.0));
    if (settings.mode == 0) {
      color *= 0.22;
      color = lerp(color, float3(0.3, 0.92, 0.8), saturate(q.w * settings.dye_strength));
    }
  }
  if (settings.smooth_boundary) {
    float2 world = float2(input.uv.x * 6 - 2, (1 - input.uv.y) * 3 - 1.5);
    float a = settings.angle * 0.01745329252;
    float q = (cos(a) * world.x - sin(a) * world.y) / 1.25 + 0.5;
    float t = (sin(a) * world.x + cos(a) * world.y) / 1.25;
    float s = clamp(q, 0, 1);
    float thickness =
        0.6 * (0.2969 * sqrt(s) - 0.126 * s - 0.3516 * s * s + 0.2843 * s * s * s - 0.1036 * s * s * s * s);
    float phi = max(abs(t) - thickness, max(-q, q - 1));
    float coverage = 1 - smoothstep(-fwidth(phi), fwidth(phi), phi);
    color = lerp(color, float3(0.82, 0.87, 0.88), coverage);
  }
  return float4(color, 1);
}
