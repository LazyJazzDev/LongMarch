float3 EvalLambertianBSDF(float3 base_color, float3 N, float3 L, out float pdf) {
  float cos_pi = max(dot(N, L), 0.0f) * INV_PI;
  pdf = cos_pi;
  return cos_pi * base_color;
}

void SampleLambertianBSDF(float3 base_color,
                          inout RandomDevice rd,
                          HitRecord hit_record,
                          out float3 eval,
                          out float3 L,
                          out float pdf) {
  SampleCosHemisphere(rd, hit_record.normal, L, pdf);
  if (dot(hit_record.geom_normal, L) > 0.0) {
    eval = pdf * base_color;
  } else {
    eval = float3(0, 0, 0);
  }
}
