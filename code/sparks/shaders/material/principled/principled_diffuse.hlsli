#pragma once
struct PrincipledDiffuseBsdf {
  Spectrum weight;
  float sample_weight;
  float3 N;
  float roughness;
};

Spectrum bsdf_principled_diffuse_compute_brdf(const PrincipledDiffuseBsdf bsdf,
                                              float3 N,
                                              float3 V,
                                              float3 L,
                                              inout float pdf) {
  const float NdotL = dot(N, L);

  if (NdotL <= 0) {
    return make_float3(0);
  }

  const float NdotV = dot(N, V);

  const float FV = schlick_fresnel(NdotV);
  const float FL = schlick_fresnel(NdotL);

  float f = 0.0f;

  /* Lambertian component. */
  f += (1.0f - 0.5f * FV) * (1.0f - 0.5f * FL);

  /* Retro-reflection component. */
  {
    /* H = normalize(L + V);  // Bisector of an angle between L and V
     * LH2 = 2 * dot(L, H)^2 = 2cos(x)^2 = cos(2x) + 1 = dot(L, V) + 1,
     * half-angle x between L and V is at most 90 deg. */
    const float LH2 = dot(L, V) + 1;
    const float RR = bsdf.roughness * LH2;
    f += RR * (FL + FV + FL * FV * (RR - 1.0f));
  }

  float value = INV_PI * NdotL * f;

  return make_float3(value);
}

Spectrum bsdf_principled_diffuse_eval(PrincipledDiffuseBsdf bsdf,
                                      const float3 I,
                                      const float3 omega_in,
                                      out float pdf) {
  const float3 N = bsdf.N;

  if (dot(N, omega_in) > 0.0f) {
    const float3 V = I;         // outgoing
    const float3 L = omega_in;  // incoming
    pdf = max(dot(N, omega_in), 0.0f) * INV_PI;
    return bsdf_principled_diffuse_compute_brdf(bsdf, N, V, L, pdf);
  } else {
    pdf = 0.0f;
    return make_float3(0);
  }
}

void bsdf_principled_diffuse_sample(PrincipledDiffuseBsdf bsdf,
                                    float3 Ng,
                                    float3 I,
                                    float randu,
                                    float randv,
                                    out Spectrum eval,
                                    out float3 omega_in,
                                    out float pdf) {
  float3 N = bsdf.N;

  sample_cos_hemisphere(N, randu, randv, omega_in, pdf);

  if (dot(Ng, omega_in) > 0) {
    eval = bsdf_principled_diffuse_compute_brdf(bsdf, N, I, omega_in, pdf);
  } else {
    pdf = 0.0f;
    eval = make_float3(0);
  }
}
