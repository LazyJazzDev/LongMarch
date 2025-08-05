#pragma once

struct PrincipledSheenBsdf {
  Spectrum weight;
  float sample_weight;
  float3 N;
  float avg_value;
};

Spectrum calculate_principled_sheen_brdf(float3 N,
                                         float3 V,
                                         float3 L,
                                         float3 H,
                                         inout float pdf) {
  float NdotL = dot(N, L);
  float NdotV = dot(N, V);

  if (NdotL < 0 || NdotV < 0) {
    pdf = 0.0f;
    return float3(0, 0, 0);
  }

  float LdotH = dot(L, H);

  float value = schlick_fresnel(LdotH) * NdotL;

  return float3(value, value, value);
}

float calculate_avg_principled_sheen_brdf(float3 N, float3 I) {
  /* To compute the average, we set the half-vector to the normal, resulting in
   * NdotI = NdotL = NdotV = LdotH */
  float NdotI = dot(N, I);
  if (NdotI < 0.0f) {
    return 0.0f;
  }

  return schlick_fresnel(NdotI) * NdotI;
}

void bsdf_principled_sheen_setup(inout PrincipledSheenBsdf bsdf) {
  bsdf.avg_value =
      calculate_avg_principled_sheen_brdf(bsdf.N, omega_v);
  bsdf.sample_weight *= bsdf.avg_value;
}

Spectrum bsdf_principled_sheen_eval(const PrincipledSheenBsdf bsdf,
                                    const float3 I,
                                    const float3 omega_in,
                                    inout float pdf) {
  const float3 N = bsdf.N;

  if (dot(N, omega_in) > 0.0f) {
    const float3 V = I;         // outgoing
    const float3 L = omega_in;  // incoming
    const float3 H = normalize(L + V);

    pdf = max(dot(N, omega_in), 0.0f) * INV_PI;
    return calculate_principled_sheen_brdf(N, V, L, H, pdf);
  } else {
    pdf = 0.0f;
    return float3(0, 0, 0);
  }
}

int bsdf_principled_sheen_sample(const PrincipledSheenBsdf bsdf,
                                 float3 Ng,
                                 float3 I,
                                 float randu,
                                 float randv,
                                 inout Spectrum eval,
                                 inout float3 omega_in,
                                 inout float pdf) {
  float3 N = bsdf.N;

  sample_cos_hemisphere(N, randu, randv, omega_in, pdf);

  if (dot(Ng, omega_in) > 0) {
    float3 H = normalize(I + omega_in);
    eval = calculate_principled_sheen_brdf(N, I, omega_in, H, pdf);
  } else {
    eval = float3(0, 0, 0);
    pdf = 0.0f;
  }
  return LABEL_REFLECT | LABEL_DIFFUSE;
}
