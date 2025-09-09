float3 sRGB2Linear(float3 srgb, float gamma) {
  float3 cutoff = step(float3(0.04045, 0.04045, 0.04045), srgb);
  float3 lower = srgb / 12.92;
  float3 higher = pow((srgb + 0.055) / 1.055, gamma);
  return lerp(lower, higher, cutoff);
}

float3 Linear2sRGB(float3 linear_color, float gamma) {
  float3 cutoff = step(float3(0.0031308, 0.0031308, 0.0031308), linear_color);
  float3 lower = linear_color * 12.92;
  float3 higher = 1.055 * pow(linear_color, 1.0 / gamma) - 0.055;
  return lerp(lower, higher, cutoff);
}
