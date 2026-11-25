struct Uniform {
  float4x4 view_projection;
  float4x4 light_projection;
  float4 light;
  float4 options;  // Screen split X, shadow resolution, reference bias, visibility only.
};

RHI_UNIFORM(0, 0, Uniform, globals);

struct Input {
  RHI_INPUT(0, float3, position);
#ifndef SHADOW_PASS
  RHI_INPUT(1, float3, normal);
  RHI_INPUT(2, float4, color);
#endif
};

#ifdef SHADOW_PASS
float4 VSMain(Input input) : SV_Position {
  return mul(globals.light_projection, float4(input.position, 1));
}

void PSMain() {
}
#else
RHI_TEXTURE2D(0, 1, float, shadow_map);
RHI_SAMPLER(0, 2, hard_sampler);
RHI_COMPARISON_SAMPLER(0, 3, pcf_sampler);

struct Varying {
  float4 position : SV_Position;
  float3 world : TEXCOORD0;
  float3 normal : TEXCOORD1;
  float4 color : TEXCOORD2;
  float4 light_position : TEXCOORD3;
};

Varying VSMain(Input input) {
  Varying output;
  output.position = mul(globals.view_projection, float4(input.position, 1));
  output.world = input.position;
  output.normal = input.normal;
  output.color = input.color;
  output.light_position = mul(globals.light_projection, float4(input.position, 1));
  return output;
}

float Visibility(float4 light_position, float diffuse, bool use_pcf) {
  float3 projected = light_position.xyz / light_position.w;
  float2 uv = projected.xy * float2(.5, -.5) + .5;
  // Receiver-plane correction keeps a wide kernel from self-shadowing a sloped floor.
  float2 dx = ddx(uv), dy = ddy(uv);
  float determinant = dx.x * dy.y - dx.y * dy.x;
  float2 depth_gradient = abs(determinant) > 1e-10 ? float2(dy.y * ddx(projected.z) - dx.y * ddy(projected.z),
                                                            dx.x * ddy(projected.z) - dy.x * ddx(projected.z)) /
                                                         determinant
                                                   : float2(0, 0);
  if (light_position.w <= 0 || any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1)
    return 1;
  float reference = projected.z - globals.options.z * (1 + 2 * (1 - diffuse));
  if (!use_pcf)
    return reference <= shadow_map.SampleLevel(hard_sampler, uv, 0) ? 1 : 0;
  float visibility = 0;
  // Each tap filters comparison results, never averages depths before comparing.
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      float2 offset = float2(x, y) / globals.options.y;
      visibility += shadow_map.SampleCmpLevelZero(pcf_sampler, uv + offset, reference + dot(depth_gradient, offset));
    }
  return visibility / 25;
}

float4 PSMain(Varying input) : SV_Target0 {
  if (input.color.w > .5)
    return float4(1, .88, .55, 1);
  float diffuse = saturate(dot(normalize(input.normal), normalize(globals.light.xyz - input.world)));
  float visibility = Visibility(input.light_position, diffuse, input.position.x >= globals.options.x);
  if (globals.options.w > .5)
    return float4(visibility.xxx, 1);
  float3 color = input.color.rgb;
  if (input.normal.y > .9 && input.world.y < .01) {
    float checker = fmod(abs(floor(input.world.x) + floor(input.world.z)), 2);
    color *= lerp(.92, 1.08, checker);
  }
  return float4(sqrt(color * (.13 + .72 * diffuse * visibility)), 1);
}
#endif
