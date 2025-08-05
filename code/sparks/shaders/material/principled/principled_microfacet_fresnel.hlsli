#pragma once


#define CLOSURE_BSDF_MICROFACET_GGX_FRESNEL_ID 0
#define CLOSURE_BSDF_MICROFACET_GGX_CLEARCOAT_ID 1
#define CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID 2
#define CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID 3

struct FresnelBsdf {
  Spectrum weight;
  float sample_weight;
  float3 N;
  float alpha_x;
  float3 T;
  float alpha_y;
  Spectrum color;
  float ior;
  Spectrum cspec0;
  Spectrum fresnel_color;
};

void bsdf_microfacet_fresnel_color(inout FresnelBsdf bsdf) {
  float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
  bsdf.fresnel_color = interpolate_fresnel_color(omega_v, bsdf.N,
                                                 bsdf.ior, F0, bsdf.cspec0);
  bsdf.sample_weight *= average(bsdf.fresnel_color);
}

Spectrum reflection_color(const FresnelBsdf bsdf, float3 L, float3 H) {
  Spectrum F = float3(1, 1, 1);
  float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
  F = interpolate_fresnel_color(L, H, bsdf.ior, F0, bsdf.cspec0);
  return F;
}

void bsdf_microfacet_ggx_fresnel_setup(inout FresnelBsdf bsdf) {
  bsdf.cspec0 = saturate(bsdf.cspec0);
  bsdf.alpha_x = saturatef(bsdf.alpha_x);
  bsdf.alpha_y = saturatef(bsdf.alpha_y);
  bsdf_microfacet_fresnel_color(bsdf);
}

Spectrum bsdf_microfacet_ggx_eval_reflect_fresnel(const FresnelBsdf bsdf,
                                                  const float3 N,
                                                  const float3 I,
                                                  const float3 omega_in,
                                                  inout float pdf,
                                                  const float alpha_x,
                                                  const float alpha_y,
                                                  const float cosNO,
                                                  const float cosNI) {
  if (!(cosNI > 0 && cosNO > 0)) {
    pdf = 0.0f;
    return float3(0, 0, 0);
  }

  /* get half vector */
  float3 m = normalize(omega_in + I);
  float alpha2 = alpha_x * alpha_y;
  float D, G1o, G1i;

  if (alpha_x == alpha_y) {
    /* isotropic
     * eq. 20: (F*G*D)/(4*in*on)
     * eq. 33: first we calculate D(m) */
    float cosThetaM = dot(N, m);
    float cosThetaM2 = cosThetaM * cosThetaM;
    float cosThetaM4 = cosThetaM2 * cosThetaM2;
    float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;

    D = alpha2 /
        (PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

    /* eq. 34: now calculate G1(i,m) and G1(o,m) */
    G1o = 2 /
          (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
    G1i = 2 /
          (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));
  } else {
    /* anisotropic */
    float3 X, Y, Z = N;
    make_orthonormals_tangent(Z, bsdf.T, X, Y);

    /* distribution */
    float3 local_m = float3(dot(X, m), dot(Y, m), dot(Z, m));
    float slope_x = -local_m.x / (local_m.z * alpha_x);
    float slope_y = -local_m.y / (local_m.z * alpha_y);
    float slope_len = 1 + slope_x * slope_x + slope_y * slope_y;

    float cosThetaM = local_m.z;
    float cosThetaM2 = cosThetaM * cosThetaM;
    float cosThetaM4 = cosThetaM2 * cosThetaM2;

    D = 1 / ((slope_len * slope_len) * PI * alpha2 * cosThetaM4);

    /* G1(i,m) and G1(o,m) */
    float tanThetaO2 = (1 - cosNO * cosNO) / (cosNO * cosNO);
    float cosPhiO = dot(I, X);
    float sinPhiO = dot(I, Y);

    float alphaO2 = (cosPhiO * cosPhiO) * (alpha_x * alpha_x) +
                    (sinPhiO * sinPhiO) * (alpha_y * alpha_y);
    alphaO2 /= cosPhiO * cosPhiO + sinPhiO * sinPhiO;

    G1o = 2 / (1 + safe_sqrtf(1 + alphaO2 * tanThetaO2));

    float tanThetaI2 = (1 - cosNI * cosNI) / (cosNI * cosNI);
    float cosPhiI = dot(omega_in, X);
    float sinPhiI = dot(omega_in, Y);

    float alphaI2 = (cosPhiI * cosPhiI) * (alpha_x * alpha_x) +
                    (sinPhiI * sinPhiI) * (alpha_y * alpha_y);
    alphaI2 /= cosPhiI * cosPhiI + sinPhiI * sinPhiI;

    G1i = 2 / (1 + safe_sqrtf(1 + alphaI2 * tanThetaI2));
  }

  float G = G1o * G1i;

  /* eq. 20 */
  float common_ = D * 0.25f / cosNO;

  Spectrum F = reflection_color(bsdf, omega_in, m);

  Spectrum out_ = F * G * common_;

  /* eq. 2 in distribution of visible normals sampling
   * `pm = Dw = G1o * dot(m, I) * D / dot(N, I);` */

  /* eq. 38 - but see also:
   * eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
   * `pdf = pm * 0.25 / dot(m, I);` */
  pdf = G1o * common_;

  return out_;
}

Spectrum bsdf_microfacet_ggx_eval_fresnel(const FresnelBsdf bsdf,
                                          const float3 I,
                                          const float3 omega_in,
                                          inout float pdf) {
  const float alpha_x = bsdf.alpha_x;
  const float alpha_y = bsdf.alpha_y;
  const float3 N = bsdf.N;
  const float cosNO = dot(N, I);
  const float cosNI = dot(N, omega_in);

  if (cosNI < 0.0f || alpha_x * alpha_y <= 1e-7f) {
    pdf = 0.0f;
    return float3(0, 0, 0);
  }

  return bsdf_microfacet_ggx_eval_reflect_fresnel(
      bsdf, N, I, omega_in, pdf, alpha_x, alpha_y, cosNO, cosNI);
}

int bsdf_microfacet_ggx_sample_fresnel(const FresnelBsdf bsdf,
                                       float3 Ng,
                                       float3 I,
                                       float randu,
                                       float randv,
                                       inout Spectrum eval,
                                       inout float3 omega_in,
                                       inout float pdf) {
  float alpha_x = bsdf.alpha_x;
  float alpha_y = bsdf.alpha_y;

  float3 N = bsdf.N;
  int label;

  float cosNO = dot(N, I);
  if (cosNO > 0) {
    float3 X, Y, Z = N;

    if (alpha_x == alpha_y)
      MakeOrthonormals(Z, X, Y);
    else
      make_orthonormals_tangent(Z, bsdf.T, X, Y);

    /* importance sampling with distribution of visible normals. vectors are
     * transformed to local space before and after */
    float3 local_I = float3(dot(X, I), dot(Y, I), cosNO);
    float3 local_m;
    float G1o;

    local_m = microfacet_sample_stretched(local_I, alpha_x, alpha_y, randu,
                                          randv, false, G1o);

    float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
    float cosThetaM = local_m.z;

    /* reflection or refraction? */
    float cosMO = dot(m, I);
    label = LABEL_REFLECT | LABEL_GLOSSY;

    if (cosMO > 0) {
      /* eq. 39 - compute actual reflected direction */
      omega_in = 2 * cosMO * m - I;

      if (dot(Ng, omega_in) > 0) {
        if (alpha_x * alpha_y <= 1e-7f) {
          /* some high number for MIS */
          pdf = 1e6f;
          eval = float3(1e6f, 1e6f, 1e6f) * reflection_color(bsdf, omega_in, m);

          label = LABEL_REFLECT | LABEL_SINGULAR;
        } else {
          /* microfacet normal is visible to this ray */
          /* eq. 33 */
          float alpha2 = alpha_x * alpha_y;
          float D, G1i;

          if (alpha_x == alpha_y) {
            /* isotropic */
            float cosThetaM2 = cosThetaM * cosThetaM;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float tanThetaM2 = 1 / (cosThetaM2)-1;

            /* eval BRDF*cosNI */
            float cosNI = dot(N, omega_in);

            D = alpha2 / (PI * cosThetaM4 * (alpha2 + tanThetaM2) *
                          (alpha2 + tanThetaM2));

            /* eq. 34: now calculate G1(i,m) */
            G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) /
                                              (cosNI * cosNI)));
          } else {
            /* anisotropic distribution */
            float3 local_m = float3(dot(X, m), dot(Y, m), dot(Z, m));
            float slope_x = -local_m.x / (local_m.z * alpha_x);
            float slope_y = -local_m.y / (local_m.z * alpha_y);
            float slope_len = 1 + slope_x * slope_x + slope_y * slope_y;

            float cosThetaM = local_m.z;
            float cosThetaM2 = cosThetaM * cosThetaM;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;

            D = 1 / ((slope_len * slope_len) * PI * alpha2 * cosThetaM4);

            /* calculate G1(i,m) */
            float cosNI = dot(N, omega_in);

            float tanThetaI2 = (1 - cosNI * cosNI) / (cosNI * cosNI);
            float cosPhiI = dot(omega_in, X);
            float sinPhiI = dot(omega_in, Y);

            float alphaI2 = (cosPhiI * cosPhiI) * (alpha_x * alpha_x) +
                            (sinPhiI * sinPhiI) * (alpha_y * alpha_y);
            alphaI2 /= cosPhiI * cosPhiI + sinPhiI * sinPhiI;

            G1i = 2 / (1 + safe_sqrtf(1 + alphaI2 * tanThetaI2));
          }

          /* see eval function for derivation */
          float common_ = (G1o * D) * 0.25f / cosNO;
          pdf = common_;

          Spectrum F = reflection_color(bsdf, omega_in, m);

          eval = G1i * common_ * F;
        }
      } else {
        eval = float3(0, 0, 0);
        pdf = 0.0f;
      }
    }
  } else {
    label = LABEL_REFLECT | LABEL_GLOSSY;
  }
  return label;
}
