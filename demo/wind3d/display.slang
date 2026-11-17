struct Uniform {
  float4x4 transform;
  float4x4 inverseTransform;
  float4 eye;
  uint4 grid;
  float4 volumeSettings;      // density, threshold [s^-1], step [m], color mode
  float4 streamlineSettings;  // viewport pixels, width pixels, opaque curvature [1/m]
  float4 curvatureSettings;   // fully transparent curvature [1/m]
};

RHI_UNIFORM(0, 0, Uniform, globals);
#ifdef WIND3D_VOLUME
RHI_BYTE_ADDRESS_BUFFER(0, 1, volume);
RHI_TEXTURE2D(0, 2, float4, sceneColor);
RHI_TEXTURE2D(0, 3, float, sceneDepth);
#endif

#if defined(WIND3D_STREAMLINES)
struct StreamInput {
  RHI_INPUT(0, float3, start);
  RHI_INPUT(1, float3, end);
  RHI_INPUT(2, float3, color);
  RHI_INPUT(3, float3, previous);
  RHI_INPUT(4, float3, next);
  RHI_INPUT(5, float2, curvatureHistory);
};

struct StreamVarying {
  float4 position : SV_Position;
  noperspective float2 local : TEXCOORD0;
  nointerpolation float segmentLength : TEXCOORD1;
  nointerpolation float3 color : COLOR0;
  nointerpolation float2 caps : TEXCOORD2;
  float curvature : TEXCOORD3;
};

StreamVarying VSStream(StreamInput input, uint id : SV_VertexID) {
  StreamVarying result;
  float4 a = mul(globals.transform, float4(input.start, 1));
  float4 b = mul(globals.transform, float4(input.end, 1));
  result.color = input.color;
  result.local = 0;
  result.curvature = 0;
  result.segmentLength = 0;
  result.caps = float2(all(input.previous == input.start) ? 1 : 0, all(input.next == input.end) ? 1 : 0);
  result.position = float4(0, 0, -1, 1);
  if (a.z < 0 && b.z < 0)
    return result;
  if (a.z < 0) {
    a = lerp(a, b, a.z / (a.z - b.z));
    result.caps.x = 1;
  }
  if (b.z < 0) {
    b = lerp(b, a, b.z / (b.z - a.z));
    result.caps.y = 1;
  }
  float2 viewport = globals.streamlineSettings.xy;
  float2 delta = (b.xy / b.w - a.xy / a.w) * viewport * 0.5;
  float segmentLength = length(delta);
  if (segmentLength < 1e-4)
    return result;
  float2 direction = delta / segmentLength;
  float2 normal = float2(-direction.y, direction.x);
  const float2 corners[6] = {float2(0, -1), float2(1, -1), float2(0, 1), float2(0, 1), float2(1, -1), float2(1, 1)};
  float2 corner = corners[id];
  result.curvature = corner.x > 0 ? input.curvatureHistory.y : input.curvatureHistory.x;
  float extent = globals.streamlineSettings.z * 0.5 + 1;
  float4 clip = corner.x > 0 ? b : a;
  float cap = corner.x > 0 ? result.caps.y : result.caps.x;
  float2 joinNormal = normal;
  if (cap == 0) {
    float4 neighbor = mul(globals.transform, float4(corner.x > 0 ? input.next : input.previous, 1));
    if (neighbor.z > 0) {
      float2 adjacent = (neighbor.xy / neighbor.w - clip.xy / clip.w) * viewport * (corner.x > 0 ? 0.5 : -0.5);
      float adjacentLength = length(adjacent);
      if (adjacentLength > 1e-4) {
        adjacent /= adjacentLength;
        float2 miter = normal + float2(-adjacent.y, adjacent.x);
        if (length(miter) > 1e-4) {
          miter = normalize(miter);
          joinNormal = miter / max(dot(miter, normal), 0.5);
        }
      }
    }
  }
  float extension = (corner.x * 2 - 1) * extent * cap;
  float2 offset = direction * extension + joinNormal * corner.y * extent;
  clip.xy += offset * 2 / viewport * clip.w;
  result.position = clip;
  result.local = float2(corner.x * segmentLength + extension, corner.y * extent);
  result.segmentLength = segmentLength;
  return result;
}

float4 PSStream(StreamVarying input) : SV_Target0 {
  float cap = max(max(-input.local.x * input.caps.x, (input.local.x - input.segmentLength) * input.caps.y), 0);
  float distance = length(float2(cap, input.local.y)) - globals.streamlineSettings.z * 0.5;
  float coverage = saturate(0.5 - distance);
  if (coverage <= 0)
    discard;
  float emphasis = saturate((input.curvature - globals.curvatureSettings.x) /
                            (globals.streamlineSettings.w - globals.curvatureSettings.x));
  if (emphasis <= 0)
    discard;
  float3 color = lerp(input.color, float3(1, 0.35, 0.06), emphasis);
  return float4(color, coverage * emphasis);
}

#elif !defined(WIND3D_VOLUME)
struct Input {
  RHI_INPUT(0, float3, position);
  RHI_INPUT(1, float3, normal);
  RHI_INPUT(2, float3, color);
};

struct Varying {
  float4 position : SV_Position;
  float3 normal : NORMAL;
  float3 color : COLOR0;
};

Varying VSMain(Input input) {
  Varying output;
  output.position = mul(globals.transform, float4(input.position, 1));
  output.normal = input.normal;
  output.color = input.color;
  return output;
}

struct SurfaceOutput {
  float4 color : SV_Target0;
  float depth : SV_Target1;
};

SurfaceOutput PSMain(Varying input) {
  float light =
      length(input.normal) < 0.1 ? 1 : 0.28 + 0.72 * abs(dot(normalize(input.normal), normalize(float3(-0.5, 1, 0.3))));
  SurfaceOutput output;
  output.color = float4(input.color * light, 1);
  output.depth = input.position.z;
  return output;
}

#else
struct VolumeVarying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

VolumeVarying VSVolume(uint id : SV_VertexID) {
  VolumeVarying result;
  result.uv = float2((id << 1) & 2, id & 2);
  result.position = float4(result.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return result;
}

float LoadVorticity(int3 p) {
  p = clamp(p, int3(0), int3(globals.grid.xyz) - 1);
  uint index = (p.z * globals.grid.y + p.y) * globals.grid.x + p.x;
  return asfloat(volume.Load(index * 4));
}

float SampleVorticity(float3 p) {
  float3 q = (p - float3(-6, -4, -6)) * (float(globals.grid.x) / 16) - 0.5;
  int3 i = int3(floor(q));
  float3 f = frac(q);
  return lerp(lerp(lerp(LoadVorticity(i), LoadVorticity(i + int3(1, 0, 0)), f.x),
                   lerp(LoadVorticity(i + int3(0, 1, 0)), LoadVorticity(i + int3(1, 1, 0)), f.x), f.y),
              lerp(lerp(LoadVorticity(i + int3(0, 0, 1)), LoadVorticity(i + int3(1, 0, 1)), f.x),
                   lerp(LoadVorticity(i + int3(0, 1, 1)), LoadVorticity(i + int3(1, 1, 1)), f.x), f.y),
              f.z);
}

float3 VorticityPalette(float value) {
  float t = saturate(log2(1 + value) / 7);
  return lerp(lerp(float3(0.04, 0.24, 0.43), float3(0.08, 0.76, 0.73), saturate(t * 2)), float3(1, 0.68, 0.25),
              saturate((t - 0.5) * 2));
}

float4 PSVolume(VolumeVarying input) : SV_Target0 {
  int2 pixel = int2(input.position.xy);
  float3 background = sceneColor.Load(int3(pixel, 0)).rgb;
  if (globals.grid.x == 0 || globals.volumeSettings.x <= 0)
    return float4(background, 1);
  float2 ndc = input.uv * float2(2, -2) + float2(-1, 1);
  float4 farPoint = mul(globals.inverseTransform, float4(ndc, 1, 1));
  float3 origin = globals.eye.xyz;
  float3 direction = normalize(farPoint.xyz / farPoint.w - origin);
  float3 inverseDirection = 1 / (direction + 1e-8);
  float3 a = (float3(-6, -4, -6) - origin) * inverseDirection;
  float3 b = (float3(10, 4, 6) - origin) * inverseDirection;
  float3 entry = min(a, b), exit = max(a, b);
  float first = max(0, max(entry.x, max(entry.y, entry.z)));
  float last = min(exit.x, min(exit.y, exit.z));
  float depth = sceneDepth.Load(int3(pixel, 0));
  if (depth < 1) {
    float4 surface = mul(globals.inverseTransform, float4(ndc, depth, 1));
    last = min(last, length(surface.xyz / surface.w - origin));
  }
  float3 color = 0;
  float transmittance = 1;
  float dt = globals.volumeSettings.z;
  [loop] for (float t = first + 0.5 * dt; t < last && transmittance > 0.012; t += dt) {
    float3 p = origin + direction * t;
    float value = SampleVorticity(p);
    float rho = max(0, value - globals.volumeSettings.y) * globals.volumeSettings.x;
    if (rho <= 0)
      continue;
    float alpha = 1 - exp(-rho * dt);
    float3 tint = globals.volumeSettings.w < 0.5 ? float3(0.75, 0.81, 0.85) : VorticityPalette(value);
    float shadow = 0;
    [unroll] for (int j = 1; j <= 3; ++j) {
      float3 q = p + float3(-0.16, 0.24, 0.12) * float(j);
      if (all(q > float3(-6, -4, -6)) && all(q < float3(10, 4, 6)))
        shadow += max(0, SampleVorticity(q) - globals.volumeSettings.y);
    }
    tint *= 0.25 + 0.75 * exp(-shadow * globals.volumeSettings.x * 0.24);
    color += transmittance * alpha * tint;
    transmittance *= 1 - alpha;
  }
  return float4(color + transmittance * background, 1);
}
#endif
