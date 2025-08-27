void SampleSpecularBSDF(float3 base_color,
float3 direction,
float3 normal,
float3 geom_normal,
out float3 eval,
out float3 L,
out float pdf) {
  L = reflect(direction, normal);
  pdf = 1e6;
  if (dot(geom_normal, L) > 0.0) {
    eval = base_color;
  } else {
    eval = float3(0, 0, 0);
  }
}
