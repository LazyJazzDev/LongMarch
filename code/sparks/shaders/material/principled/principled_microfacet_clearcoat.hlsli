#pragma once

#define CLOSURE_BSDF_MICROFACET_GGX_FRESNEL_ID 0
#define CLOSURE_BSDF_MICROFACET_GGX_CLEARCOAT_ID 1
#define CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID 2
#define CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID 3

struct ClearcoatBsdf {
  Spectrum weight;
  float sample_weight;
  float3 N;
  float alpha;
  Spectrum cspec0;
  float ior;
  Spectrum fresnel_color;
  float clearcoat;
};

void bsdf_microfacet_fresnel_color(inout ClearcoatBsdf bsdf) {
  float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
  bsdf.fresnel_color = interpolate_fresnel_color(omega_v, bsdf.N,
                                                 bsdf.ior, F0, bsdf.cspec0) *
                       0.25f * bsdf.clearcoat;
  bsdf.sample_weight *= average(bsdf.fresnel_color);
}

Spectrum reflection_color(const ClearcoatBsdf bsdf, float3 L, float3 H) {
  Spectrum F = make_float3(1);
  float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
  F = interpolate_fresnel_color(L, H, bsdf.ior, F0, bsdf.cspec0);
  return F;
}

void bsdf_microfacet_ggx_clearcoat_setup(inout ClearcoatBsdf bsdf) {
  bsdf.cspec0 = saturate(bsdf.cspec0);
  bsdf.alpha = saturatef(bsdf.alpha);
  bsdf_microfacet_fresnel_color(bsdf);
}

Spectrum bsdf_microfacet_ggx_eval_reflect_clearcoat(const ClearcoatBsdf bsdf,
                                                    const float3 N,
                                                    const float3 I,
                                                    const float3 omega_in,
                                                    inout float pdf,
                                                    const float alpha,
                                                    const float cosNO,
                                                    const float cosNI) {
  if (!(cosNI > 0 && cosNO > 0)) {
    pdf = 0.0f;
    return make_float3(0);
  }

  /* get half vector */
  float3 m = normalize(omega_in + I);
  float alpha2 = alpha * alpha;
  float D, G1o, G1i;

  /* isotropic
   * eq. 20: (F*G*D)/(4*in*on)
   * eq. 33: first we calculate D(m) */
  float cosThetaM = dot(N, m);
  float cosThetaM2 = cosThetaM * cosThetaM;
  float cosThetaM4 = cosThetaM2 * cosThetaM2;
  float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;

  /* use GTR1 for clearcoat */
  D = D_GTR1(cosThetaM, bsdf.alpha);

  /* the alpha value for clearcoat is a fixed 0.25 => alpha2 = 0.25 * 0.25
   */
  alpha2 = 0.0625f;

  /* eq. 34: now calculate G1(i,m) and G1(o,m) */
  G1o =
      2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
  G1i =
      2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

  float G = G1o * G1i;

  /* eq. 20 */
  float common_ = D * 0.25f / cosNO;

  Spectrum F = reflection_color(bsdf, omega_in, m) * 0.25f * bsdf.clearcoat;

  Spectrum out_ = F * G * common_;

  /* eq. 2 in distribution of visible normals sampling
   * `pm = Dw = G1o * dot(m, I) * D / dot(N, I);` */

  /* eq. 38 - but see also:
   * eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
   * `pdf = pm * 0.25 / dot(m, I);` */
  pdf = G1o * common_;

  return out_;
}

Spectrum bsdf_microfacet_ggx_eval_clearcoat(const ClearcoatBsdf bsdf,
                                            const float3 I,
                                            const float3 omega_in,
                                            inout float pdf) {
  const float alpha = bsdf.alpha;
  const float3 N = bsdf.N;
  const float cosNO = dot(N, I);
  const float cosNI = dot(N, omega_in);

  if (cosNI < 0.0f || alpha * alpha <= 1e-7f) {
    pdf = 0.0f;
    return make_float3(0.0);
  }

  return bsdf_microfacet_ggx_eval_reflect_clearcoat(bsdf, N, I, omega_in, pdf,
                                                    alpha, cosNO, cosNI);
}

int bsdf_microfacet_ggx_sample_clearcoat(const ClearcoatBsdf bsdf,
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
    float3 local_I = make_float3(dot(X, I), dot(Y, I), cosNO);
    float3 local_m;
    float G1o;

    local_m = microfacet_sample_stretched(local_I, alpha, alpha, randu, randv,
                                          false, G1o);

    float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
    float cosThetaM = local_m.z;

    /* reflection or refraction? */
    float cosMO = dot(m, I);
    label = LABEL_REFLECT | LABEL_GLOSSY;

    if (cosMO > 0) {
      /* eq. 39 - compute actual reflected direction */
      omega_in = 2 * cosMO * m - I;

      if (dot(Ng, omega_in) > 0) {
        if (alpha * alpha <= 1e-7f) {
          /* some high number for MIS */
          pdf = 1e6f;
          eval = make_float3(1e6f) * reflection_color(bsdf, omega_in, m);

          label = LABEL_REFLECT | LABEL_SINGULAR;
        } else {
          /* microfacet normal is visible to this ray */
          /* eq. 33 */
          float alpha2 = alpha * alpha;
          float D, G1i;

          /* isotropic */
          float cosThetaM2 = cosThetaM * cosThetaM;
          float cosThetaM4 = cosThetaM2 * cosThetaM2;
          float tanThetaM2 = 1 / (cosThetaM2)-1;

          /* eval BRDF*cosNI */
          float cosNI = dot(N, omega_in);

          /* use GTR1 for clearcoat */
          D = D_GTR1(cosThetaM, bsdf.alpha);

          /* the alpha value for clearcoat is a fixed 0.25 => alpha2 =
           * 0.25 * 0.25 */
          alpha2 = 0.0625f;

          /* recalculate G1o */
          G1o = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) /
                                            (cosNO * cosNO)));

          /* eq. 34: now calculate G1(i,m) */
          G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) /
                                            (cosNI * cosNI)));

          /* see eval function for derivation */
          float common_ = (G1o * D) * 0.25f / cosNO;
          pdf = common_;

          Spectrum F = reflection_color(bsdf, omega_in, m);

          eval = G1i * common_ * F;
        }

        eval *= 0.25f * bsdf.clearcoat;
      } else {
        eval = make_float3(0);
        pdf = 0.0f;
      }
    }
  } else {
    label = LABEL_REFLECT | LABEL_GLOSSY;
  }
  return label;
}
