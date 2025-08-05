#pragma once

#define CLOSURE_BSDF_MICROFACET_GGX_FRESNEL_ID 0
#define CLOSURE_BSDF_MICROFACET_GGX_CLEARCOAT_ID 1
#define CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID 2
#define CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID 3

struct RefractionBsdf {
  Spectrum weight;
  float sample_weight;
  float3 N;
  float alpha, ior;
};

Spectrum bsdf_microfacet_ggx_eval_transmit_refraction(const RefractionBsdf bsdf,
                                                      const float3 N,
                                                      const float3 I,
                                                      const float3 omega_in,
                                                      out float pdf,
                                                      const float alpha,
                                                      const float cosNO,
                                                      const float cosNI) {
  if (cosNO <= 0 || cosNI >= 0) {
    pdf = 0.0f;
    return float3(0, 0, 0); /* vectors on same side -- not possible */
  }
  /* compute half-vector of the refraction (eq. 16) */
  float m_eta = bsdf.ior;
  float3 ht = -(m_eta * omega_in + I);
  float3 Ht = normalize(ht);
  float cosHO = dot(Ht, I);
  float cosHI = dot(Ht, omega_in);

  float D, G1o, G1i;

  /* eq. 33: first we calculate D(m) with m=Ht: */
  float alpha2 = alpha * alpha;
  float cosThetaM = dot(N, Ht);
  float cosThetaM2 = cosThetaM * cosThetaM;
  float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
  float cosThetaM4 = cosThetaM2 * cosThetaM2;
  D = alpha2 /
      (PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

  /* eq. 34: now calculate G1(i,m) and G1(o,m) */
  G1o =
      2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
  G1i =
      2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

  float G = G1o * G1i;

  /* probability */
  float Ht2 = dot(ht, ht);

  /* eq. 2 in distribution of visible normals sampling
   * pm = Dw = G1o * dot(m, I) * D / dot(N, I); */

  /* out = fabsf(cosHI * cosHO) * (m_eta * m_eta) * G * D / (cosNO * Ht2)
   * pdf = pm * (m_eta * m_eta) * fabsf(cosHI) / Ht2 */
  float common_ = D * (m_eta * m_eta) / (cosNO * Ht2);
  float out_ = G * abs(cosHI * cosHO) * common_;
  pdf = G1o * abs(cosHO * cosHI) * common_;

  return float3(out_, out_, out_);
}

Spectrum bsdf_microfacet_ggx_eval_refraction(const RefractionBsdf bsdf,
                                             const float3 I,
                                             const float3 omega_in,
                                             out float pdf) {
  const float alpha = bsdf.alpha;
  const float3 N = bsdf.N;
  const float cosNO = dot(N, I);
  const float cosNI = dot(N, omega_in);

  if (!(cosNI < 0.0f) || alpha * alpha <= 1e-7f) {
    pdf = 0.0f;
    return float3(0, 0, 0);
  }

  return bsdf_microfacet_ggx_eval_transmit_refraction(bsdf, N, I, omega_in, pdf,
                                                      alpha, cosNO, cosNI);
}

int bsdf_microfacet_ggx_sample_refraction(const RefractionBsdf bsdf,
                                          float3 Ng,
                                          float3 I,
                                          float randu,
                                          float randv,
                                          inout Spectrum eval,
                                          inout float3 omega_in,
                                          inout float pdf) {
  float alpha = bsdf.alpha;

  float3 N = bsdf.N;
  int label;

  float cosNO = dot(N, I);
  if (cosNO > 0) {
    float3 X, Y, Z = N;

    MakeOrthonormals(Z, X, Y);

    /* importance sampling with distribution of visible normals. vectors are
     * transformed to local space before and after */
    float3 local_I = float3(dot(X, I), dot(Y, I), cosNO);
    float3 local_m;
    float G1o;

    local_m = microfacet_sample_stretched(local_I, alpha, alpha, randu, randv,
                                          false, G1o);

    float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
    float cosThetaM = local_m.z;

    label = LABEL_TRANSMIT | LABEL_GLOSSY;

    /* CAUTION: the i and o variables are inverted relative to the paper
     * eq. 39 - compute actual refractive direction */
    float3 R, T;
    float m_eta = bsdf.ior, fresnel;
    bool inside;

    fresnel = fresnel_dielectric(m_eta, m, I, R, T, inside);

    if (!inside && fresnel != 1.0f) {
      omega_in = T;

      if (alpha * alpha <= 1e-7f || abs(m_eta - 1.0f) < 1e-4f) {
        /* some high number for MIS */
        pdf = 1e6f;
        eval = float3(1e6f, 1e6f, 1e6f);
        label = LABEL_TRANSMIT | LABEL_SINGULAR;
      } else {
        /* eq. 33 */
        float alpha2 = alpha * alpha;
        float cosThetaM2 = cosThetaM * cosThetaM;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        float tanThetaM2 = 1 / (cosThetaM2)-1;
        float D = alpha2 / (PI * cosThetaM4 * (alpha2 + tanThetaM2) *
                            (alpha2 + tanThetaM2));

        /* eval BRDF*cosNI */
        float cosNI = dot(N, omega_in);

        /* eq. 34: now calculate G1(i,m) */
        float G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) /
                                                (cosNI * cosNI)));

        /* eq. 21 */
        float cosHI = dot(m, omega_in);
        float cosHO = dot(m, I);
        float Ht2 = m_eta * cosHI + cosHO;
        Ht2 *= Ht2;

        /* see eval function for derivation */
        float common_ = (G1o * D) * (m_eta * m_eta) / (cosNO * Ht2);
        float out_ = G1i * abs(cosHI * cosHO) * common_;
        pdf = cosHO * abs(cosHI) * common_;

        eval = float3(out_, out_, out_);
      }
    } else {
      eval = float3(0, 0, 0);
      pdf = 0.0f;
    }
  } else {
    label = LABEL_TRANSMIT | LABEL_GLOSSY;
  }
  return label;
}
