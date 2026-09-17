#pragma once
// Portable port of the principled material closures:
//   code/sparkium/shaders/bsdf/principled_material.hlsli
//   code/sparkium/shaders/bsdf/principled_bsdf.hlsli
//   code/sparkium/shaders/bsdf/principled_diffuse.hlsli
//   code/sparkium/shaders/bsdf/principled_sheen.hlsli
//   code/sparkium/shaders/bsdf/principled_microfacet.hlsli
//   code/sparkium/shaders/bsdf/principled_microfacet_fresnel.hlsli
//   code/sparkium/shaders/bsdf/principled_microfacet_clearcoat.hlsli
//   code/sparkium/shaders/bsdf/principled_microfacet_refraction.hlsli
//
// The HLSL declares these as member functions of a `PrincipledMaterial` class
// that reads `omega_v` and the material parameters from the enclosing class;
// the port keeps exactly that structure so the two files can be diffed.

#include "sparkium/backends/core/bsdf_util.h"

namespace sparkium::backends {

using device::make_float3;
using device::saturatef;

struct PrincipledDiffuseBsdf {
  float3 weight;
  float sample_weight;
  float3 N;
  float roughness;
};

struct PrincipledSheenBsdf {
  float3 weight;
  float sample_weight;
  float3 N;
  float avg_value;
};

struct FresnelBsdf {
  float3 weight;
  float sample_weight;
  float3 N;
  float alpha_x;
  float3 T;
  float alpha_y;
  float3 color;
  float ior;
  float3 cspec0;
  float3 fresnel_color;
};

struct ClearcoatBsdf {
  float3 weight;
  float sample_weight;
  float3 N;
  float alpha;
  float3 cspec0;
  float ior;
  float3 fresnel_color;
  float clearcoat;
};

struct RefractionBsdf {
  float3 weight;
  float sample_weight;
  float3 N;
  float alpha, ior;
};

struct PrincipledMaterial {
  HitRecord hit_record;
  float3 omega_v;
  PrincipledDiffuseBsdf diffuse_closure;
  FresnelBsdf microfacet_closure;
  FresnelBsdf microfacet_bsdf_reflect_closure;
  RefractionBsdf microfacet_bsdf_refract_closure;
  ClearcoatBsdf microfacet_clearcoat_closure;
  PrincipledSheenBsdf sheen_closure;

  float3 base_color;

  float3 subsurface_color;
  float subsurface;

  float3 subsurface_radius;
  float metallic;

  float specular;
  float specular_tint;
  float roughness;
  float anisotropic;

  float anisotropic_rotation;
  float sheen;
  float sheen_tint;
  float clearcoat;

  float clearcoat_roughness;
  float ior;
  float transmission;
  float transmission_roughness;

  static const int CLOSURE_COUNT = 6;

  // -------------------------------------------------------------------------
  // principled_diffuse.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD float3 bsdf_principled_diffuse_compute_brdf(const PrincipledDiffuseBsdf &bsdf,
                                                          const float3 &N,
                                                          const float3 &V,
                                                          const float3 &L,
                                                          float &pdf) const {
    const float NdotL = device::dot(N, L);

    if (NdotL <= 0) {
      return make_float3(0);
    }

    const float NdotV = device::dot(N, V);

    const float FV = schlick_fresnel(NdotV);
    const float FL = schlick_fresnel(NdotL);

    float f = 0.0f;

    /* Lambertian component. */
    f += (1.0f - 0.5f * FV) * (1.0f - 0.5f * FL);

    /* Retro-reflection component. */
    {
      const float LH2 = device::dot(L, V) + 1;
      const float RR = bsdf.roughness * LH2;
      f += RR * (FL + FV + FL * FV * (RR - 1.0f));
    }

    float value = SPARKIUM_INV_PI * NdotL * f;

    return make_float3(value);
  }

  SPARKIUM_HD float3 bsdf_principled_diffuse_eval(const PrincipledDiffuseBsdf &bsdf,
                                                  const float3 &I,
                                                  const float3 &omega_in,
                                                  float &pdf) const {
    const float3 N = bsdf.N;

    if (device::dot(N, omega_in) > 0.0f) {
      const float3 V = I;         // outgoing
      const float3 L = omega_in;  // incoming
      pdf = device::max(device::dot(N, omega_in), 0.0f) * SPARKIUM_INV_PI;
      return bsdf_principled_diffuse_compute_brdf(bsdf, N, V, L, pdf);
    } else {
      pdf = 0.0f;
      return make_float3(0);
    }
  }

  SPARKIUM_HD void bsdf_principled_diffuse_sample(const PrincipledDiffuseBsdf &bsdf,
                                                  const float3 &Ng,
                                                  const float3 &I,
                                                  float randu,
                                                  float randv,
                                                  float3 &eval,
                                                  float3 &omega_in,
                                                  float &pdf) const {
    float3 N = bsdf.N;

    sample_cos_hemisphere(N, randu, randv, omega_in, pdf);

    if (device::dot(Ng, omega_in) > 0) {
      eval = bsdf_principled_diffuse_compute_brdf(bsdf, N, I, omega_in, pdf);
    } else {
      pdf = 0.0f;
      eval = make_float3(0);
    }
  }

  // -------------------------------------------------------------------------
  // principled_sheen.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD float3 calculate_principled_sheen_brdf(const float3 &N,
                                                     const float3 &V,
                                                     const float3 &L,
                                                     const float3 &H,
                                                     float &pdf) const {
    float NdotL = device::dot(N, L);
    float NdotV = device::dot(N, V);

    if (NdotL < 0 || NdotV < 0) {
      pdf = 0.0f;
      return make_float3(0);
    }

    float LdotH = device::dot(L, H);

    float value = schlick_fresnel(LdotH) * NdotL;

    return make_float3(value);
  }

  SPARKIUM_HD float calculate_avg_principled_sheen_brdf(const float3 &N, const float3 &I) const {
    /* To compute the average, we set the half-vector to the normal, resulting in
     * NdotI = NdotL = NdotV = LdotH */
    float NdotI = device::dot(N, I);
    if (NdotI < 0.0f) {
      return 0.0f;
    }

    return schlick_fresnel(NdotI) * NdotI;
  }

  SPARKIUM_HD void bsdf_principled_sheen_setup(PrincipledSheenBsdf &bsdf) const {
    bsdf.avg_value = calculate_avg_principled_sheen_brdf(bsdf.N, omega_v);
    bsdf.sample_weight *= bsdf.avg_value;
  }

  SPARKIUM_HD float3 bsdf_principled_sheen_eval(const PrincipledSheenBsdf &bsdf,
                                                const float3 &I,
                                                const float3 &omega_in,
                                                float &pdf) const {
    const float3 N = bsdf.N;

    if (device::dot(N, omega_in) > 0.0f) {
      const float3 V = I;         // outgoing
      const float3 L = omega_in;  // incoming
      const float3 H = device::normalize(L + V);

      pdf = device::max(device::dot(N, omega_in), 0.0f) * SPARKIUM_INV_PI;
      return calculate_principled_sheen_brdf(N, V, L, H, pdf);
    } else {
      pdf = 0.0f;
      return make_float3(0);
    }
  }

  SPARKIUM_HD int bsdf_principled_sheen_sample(const PrincipledSheenBsdf &bsdf,
                                               const float3 &Ng,
                                               const float3 &I,
                                               float randu,
                                               float randv,
                                               float3 &eval,
                                               float3 &omega_in,
                                               float &pdf) const {
    float3 N = bsdf.N;

    sample_cos_hemisphere(N, randu, randv, omega_in, pdf);

    if (device::dot(Ng, omega_in) > 0) {
      float3 H = device::normalize(I + omega_in);
      eval = calculate_principled_sheen_brdf(N, I, omega_in, H, pdf);
    } else {
      eval = make_float3(0.0);
      pdf = 0.0f;
    }
    return SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_DIFFUSE;
  }

  // -------------------------------------------------------------------------
  // principled_microfacet.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD void microfacet_ggx_sample_slopes(const float cos_theta_i,
                                                const float sin_theta_i,
                                                float randu,
                                                float randv,
                                                float &slope_x,
                                                float &slope_y,
                                                float &G1i) const {
    /* special case (normal incidence) */
    if (cos_theta_i >= 0.99999f) {
      const float r = sqrtf(randu / (1.0f - randu));
      const float phi = 2.0 * SPARKIUM_PI * randv;
      slope_x = r * cosf(phi);
      slope_y = r * sinf(phi);
      G1i = 1.0f;

      return;
    }

    /* precomputations */
    const float tan_theta_i = sin_theta_i / cos_theta_i;
    const float G1_inv = 0.5f * (1.0f + safe_sqrtf(1.0f + tan_theta_i * tan_theta_i));

    G1i = 1.0f / G1_inv;

    /* sample slope_x */
    const float A = 2.0f * randu * G1_inv - 1.0f;
    const float AA = A * A;
    const float tmp = 1.0f / (AA - 1.0f);
    const float B = tan_theta_i;
    const float BB = B * B;
    const float D = safe_sqrtf(BB * (tmp * tmp) - (AA - BB) * tmp);
    const float slope_x_1 = B * tmp - D;
    const float slope_x_2 = B * tmp + D;
    slope_x = (A < 0.0f || slope_x_2 * tan_theta_i > 1.0f) ? slope_x_1 : slope_x_2;

    /* sample slope_y */
    float S;

    if (randv > 0.5f) {
      S = 1.0f;
      randv = 2.0f * (randv - 0.5f);
    } else {
      S = -1.0f;
      randv = 2.0f * (0.5f - randv);
    }

    const float z = (randv * (randv * (randv * 0.27385f - 0.73369f) + 0.46341f)) /
                    (randv * (randv * (randv * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    slope_y = S * z * safe_sqrtf(1.0f + (slope_x) * (slope_x));
  }

  SPARKIUM_HD float3 microfacet_sample_stretched(const float3 &omega_i,
                                                 const float alpha_x,
                                                 const float alpha_y,
                                                 const float randu,
                                                 const float randv,
                                                 bool beckmann,
                                                 float &G1i) const {
    /* 1. stretch omega_i */
    float3 omega_i_ = make_float3(alpha_x * omega_i.x, alpha_y * omega_i.y, omega_i.z);
    omega_i_ = device::normalize(omega_i_);

    /* get polar coordinates of omega_i_ */
    float costheta_ = 1.0f;
    float sintheta_ = 0.0f;
    float cosphi_ = 1.0f;
    float sinphi_ = 0.0f;

    if (omega_i_.z < 0.99999f) {
      costheta_ = omega_i_.z;
      sintheta_ = safe_sqrtf(1.0f - costheta_ * costheta_);

      float invlen = 1.0f / sintheta_;
      cosphi_ = omega_i_.x * invlen;
      sinphi_ = omega_i_.y * invlen;
    }

    /* 2. sample P22_{omega_i}(x_slope, y_slope, 1, 1) */
    float slope_x, slope_y;

    if (beckmann) {
      microfacet_beckmann_sample_slopes(costheta_, sintheta_, randu, randv, slope_x, slope_y, G1i);
    } else {
      microfacet_ggx_sample_slopes(costheta_, sintheta_, randu, randv, slope_x, slope_y, G1i);
    }

    /* 3. rotate */
    float tmp = cosphi_ * slope_x - sinphi_ * slope_y;
    slope_y = sinphi_ * slope_x + cosphi_ * slope_y;
    slope_x = tmp;

    /* 4. unstretch */
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    /* 5. compute normal */
    return device::normalize(make_float3(-slope_x, -slope_y, 1.0f));
  }

  SPARKIUM_HD void microfacet_beckmann_sample_slopes(const float cos_theta_i,
                                                     const float sin_theta_i,
                                                     float randu,
                                                     float randv,
                                                     float &slope_x,
                                                     float &slope_y,
                                                     float &G1i) const {
    /* special case (normal incidence) */
    if (cos_theta_i >= 0.99999f) {
      const float r = sqrtf(-logf(randu));
      const float phi = 2.0 * SPARKIUM_PI * randv;
      slope_x = r * cosf(phi);
      slope_y = r * sinf(phi);
      G1i = 1.0f;
      return;
    }

    /* precomputations */
    const float tan_theta_i = sin_theta_i / cos_theta_i;
    const float inv_a = tan_theta_i;
    const float cot_theta_i = 1.0f / tan_theta_i;
    const float erf_a = fast_erff(cot_theta_i);
    const float exp_a2 = expf(-cot_theta_i * cot_theta_i);
    const float SQRT_PI_INV = 0.56418958354f;
    const float Lambda = 0.5f * (erf_a - 1.0f) + (0.5f * SQRT_PI_INV) * (exp_a2 * inv_a);
    const float G1 = 1.0f / (1.0f + Lambda); /* masking */

    G1i = G1;

    float K = tan_theta_i * SQRT_PI_INV;
    float y_approx = randu * (1.0f + erf_a + K * (1 - erf_a * erf_a));
    float y_exact = randu * (1.0f + erf_a + K * exp_a2);
    float b = K > 0 ? (0.5f - sqrtf(K * (K - y_approx + 1.0f) + 0.25f)) / K : y_approx - 1.0f;

    /* Perform newton step to refine toward the true root. */
    float inv_erf = fast_ierff(b);
    float value = 1.0f + b + K * expf(-inv_erf * inv_erf) - y_exact;
    /* Check if we are close enough already,
     * this also avoids NaNs as we get close to the root.
     */
    if (fabsf(value) > 1e-6f) {
      b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 1. */
      inv_erf = fast_ierff(b);
      value = 1.0f + b + K * expf(-inv_erf * inv_erf) - y_exact;
      b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 2. */
      /* Compute the slope from the refined value. */
      slope_x = fast_ierff(b);
    } else {
      /* We are close enough already. */
      slope_x = inv_erf;
    }
    slope_y = fast_ierff(2.0f * randv - 1.0f);
  }

  // -------------------------------------------------------------------------
  // principled_microfacet_fresnel.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD void bsdf_microfacet_fresnel_color(FresnelBsdf &bsdf) const {
    float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
    bsdf.fresnel_color = interpolate_fresnel_color(omega_v, bsdf.N, bsdf.ior, F0, bsdf.cspec0);
    bsdf.sample_weight *= average(bsdf.fresnel_color);
  }

  SPARKIUM_HD float3 reflection_color(const FresnelBsdf &bsdf, const float3 &L, const float3 &H) const {
    float3 F = make_float3(1);
    float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
    F = interpolate_fresnel_color(L, H, bsdf.ior, F0, bsdf.cspec0);
    return F;
  }

  SPARKIUM_HD void bsdf_microfacet_ggx_fresnel_setup(FresnelBsdf &bsdf) const {
    bsdf.cspec0 = device::saturate(bsdf.cspec0);
    bsdf.alpha_x = saturatef(bsdf.alpha_x);
    bsdf.alpha_y = saturatef(bsdf.alpha_y);
    bsdf_microfacet_fresnel_color(bsdf);
  }

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_reflect_fresnel(const FresnelBsdf &bsdf,
                                                              const float3 &N,
                                                              const float3 &I,
                                                              const float3 &omega_in,
                                                              float &pdf,
                                                              const float alpha_x,
                                                              const float alpha_y,
                                                              const float cosNO,
                                                              const float cosNI) const {
    if (!(cosNI > 0 && cosNO > 0)) {
      pdf = 0.0f;
      return make_float3(0);
    }

    /* get half vector */
    float3 m = device::normalize(omega_in + I);
    float alpha2 = alpha_x * alpha_y;
    float D, G1o, G1i;

    if (alpha_x == alpha_y) {
      /* isotropic
       * eq. 20: (F*G*D)/(4*in*on)
       * eq. 33: first we calculate D(m) */
      float cosThetaM = device::dot(N, m);
      float cosThetaM2 = cosThetaM * cosThetaM;
      float cosThetaM4 = cosThetaM2 * cosThetaM2;
      float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;

      D = alpha2 / (SPARKIUM_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

      /* eq. 34: now calculate G1(i,m) and G1(o,m) */
      G1o = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
      G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));
    } else {
      /* anisotropic */
      float3 X, Y, Z = N;
      make_orthonormals_tangent(Z, bsdf.T, X, Y);

      /* distribution */
      float3 local_m = make_float3(device::dot(X, m), device::dot(Y, m), device::dot(Z, m));
      float slope_x = -local_m.x / (local_m.z * alpha_x);
      float slope_y = -local_m.y / (local_m.z * alpha_y);
      float slope_len = 1 + slope_x * slope_x + slope_y * slope_y;

      float cosThetaM = local_m.z;
      float cosThetaM2 = cosThetaM * cosThetaM;
      float cosThetaM4 = cosThetaM2 * cosThetaM2;

      D = 1 / ((slope_len * slope_len) * SPARKIUM_PI * alpha2 * cosThetaM4);

      /* G1(i,m) and G1(o,m) */
      float tanThetaO2 = (1 - cosNO * cosNO) / (cosNO * cosNO);
      float cosPhiO = device::dot(I, X);
      float sinPhiO = device::dot(I, Y);

      float alphaO2 = (cosPhiO * cosPhiO) * (alpha_x * alpha_x) + (sinPhiO * sinPhiO) * (alpha_y * alpha_y);
      alphaO2 /= cosPhiO * cosPhiO + sinPhiO * sinPhiO;

      G1o = 2 / (1 + safe_sqrtf(1 + alphaO2 * tanThetaO2));

      float tanThetaI2 = (1 - cosNI * cosNI) / (cosNI * cosNI);
      float cosPhiI = device::dot(omega_in, X);
      float sinPhiI = device::dot(omega_in, Y);

      float alphaI2 = (cosPhiI * cosPhiI) * (alpha_x * alpha_x) + (sinPhiI * sinPhiI) * (alpha_y * alpha_y);
      alphaI2 /= cosPhiI * cosPhiI + sinPhiI * sinPhiI;

      G1i = 2 / (1 + safe_sqrtf(1 + alphaI2 * tanThetaI2));
    }

    float G = G1o * G1i;

    /* eq. 20 */
    float common_ = D * 0.25f / cosNO;

    float3 F = reflection_color(bsdf, omega_in, m);

    float3 out_ = F * G * common_;

    pdf = G1o * common_;

    return out_;
  }

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_fresnel(const FresnelBsdf &bsdf,
                                                      const float3 &I,
                                                      const float3 &omega_in,
                                                      float &pdf) const {
    const float alpha_x = bsdf.alpha_x;
    const float alpha_y = bsdf.alpha_y;
    const float3 N = bsdf.N;
    const float cosNO = device::dot(N, I);
    const float cosNI = device::dot(N, omega_in);

    if (cosNI < 0.0f || alpha_x * alpha_y <= 1e-7f) {
      pdf = 0.0f;
      return make_float3(0.0);
    }

    return bsdf_microfacet_ggx_eval_reflect_fresnel(bsdf, N, I, omega_in, pdf, alpha_x, alpha_y, cosNO, cosNI);
  }

  SPARKIUM_HD int bsdf_microfacet_ggx_sample_fresnel(const FresnelBsdf &bsdf,
                                                     const float3 &Ng,
                                                     const float3 &I,
                                                     float randu,
                                                     float randv,
                                                     float3 &eval,
                                                     float3 &omega_in,
                                                     float &pdf) const {
    float alpha_x = bsdf.alpha_x;
    float alpha_y = bsdf.alpha_y;

    float3 N = bsdf.N;
    int label;

    float cosNO = device::dot(N, I);
    if (cosNO > 0) {
      float3 X, Y, Z = N;

      if (alpha_x == alpha_y)
        MakeOrthonormals(Z, X, Y);
      else
        make_orthonormals_tangent(Z, bsdf.T, X, Y);

      /* importance sampling with distribution of visible normals. vectors are
       * transformed to local space before and after */
      float3 local_I = make_float3(device::dot(X, I), device::dot(Y, I), cosNO);
      float3 local_m;
      float G1o;

      local_m = microfacet_sample_stretched(local_I, alpha_x, alpha_y, randu, randv, false, G1o);

      float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
      float cosThetaM = local_m.z;

      /* reflection or refraction? */
      float cosMO = device::dot(m, I);
      label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_GLOSSY;

      if (cosMO > 0) {
        /* eq. 39 - compute actual reflected direction */
        omega_in = 2 * cosMO * m - I;

        if (device::dot(Ng, omega_in) > 0) {
          if (alpha_x * alpha_y <= 1e-7f) {
            /* some high number for MIS */
            pdf = 1e6f;
            eval = make_float3(1e6f) * reflection_color(bsdf, omega_in, m);

            label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_SINGULAR;
          } else {
            /* microfacet normal is visible to this ray */
            /* eq. 33 */
            float alpha2 = alpha_x * alpha_y;
            float D, G1i;

            if (alpha_x == alpha_y) {
              /* isotropic */
              float cosThetaM2 = cosThetaM * cosThetaM;
              float cosThetaM4 = cosThetaM2 * cosThetaM2;
              float tanThetaM2 = 1 / (cosThetaM2) - 1;

              /* eval BRDF*cosNI */
              float cosNI = device::dot(N, omega_in);

              D = alpha2 / (SPARKIUM_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

              /* eq. 34: now calculate G1(i,m) */
              G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));
            } else {
              /* anisotropic distribution */
              float3 local_m2 = make_float3(device::dot(X, m), device::dot(Y, m), device::dot(Z, m));
              float slope_x = -local_m2.x / (local_m2.z * alpha_x);
              float slope_y = -local_m2.y / (local_m2.z * alpha_y);
              float slope_len = 1 + slope_x * slope_x + slope_y * slope_y;

              float cosThetaM_ = local_m2.z;
              float cosThetaM2_ = cosThetaM_ * cosThetaM_;
              float cosThetaM4_ = cosThetaM2_ * cosThetaM2_;

              D = 1 / ((slope_len * slope_len) * SPARKIUM_PI * alpha2 * cosThetaM4_);

              /* calculate G1(i,m) */
              float cosNI = device::dot(N, omega_in);

              float tanThetaI2 = (1 - cosNI * cosNI) / (cosNI * cosNI);
              float cosPhiI = device::dot(omega_in, X);
              float sinPhiI = device::dot(omega_in, Y);

              float alphaI2 = (cosPhiI * cosPhiI) * (alpha_x * alpha_x) + (sinPhiI * sinPhiI) * (alpha_y * alpha_y);
              alphaI2 /= cosPhiI * cosPhiI + sinPhiI * sinPhiI;

              G1i = 2 / (1 + safe_sqrtf(1 + alphaI2 * tanThetaI2));
            }

            /* see eval function for derivation */
            float common_ = (G1o * D) * 0.25f / cosNO;
            pdf = common_;

            float3 F = reflection_color(bsdf, omega_in, m);

            eval = G1i * common_ * F;
          }
        } else {
          eval = make_float3(0);
          pdf = 0.0f;
        }
      }
    } else {
      label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_GLOSSY;
    }
    return label;
  }

  // -------------------------------------------------------------------------
  // principled_microfacet_clearcoat.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD void bsdf_microfacet_fresnel_color(ClearcoatBsdf &bsdf) const {
    float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
    bsdf.fresnel_color =
        interpolate_fresnel_color(omega_v, bsdf.N, bsdf.ior, F0, bsdf.cspec0) * 0.25f * bsdf.clearcoat;
    bsdf.sample_weight *= average(bsdf.fresnel_color);
  }

  SPARKIUM_HD float3 reflection_color(const ClearcoatBsdf &bsdf, const float3 &L, const float3 &H) const {
    float3 F = make_float3(1);
    float F0 = fresnel_dielectric_cos(1.0f, bsdf.ior);
    F = interpolate_fresnel_color(L, H, bsdf.ior, F0, bsdf.cspec0);
    return F;
  }

  SPARKIUM_HD void bsdf_microfacet_ggx_clearcoat_setup(ClearcoatBsdf &bsdf) const {
    bsdf.cspec0 = device::saturate(bsdf.cspec0);
    bsdf.alpha = saturatef(bsdf.alpha);
    bsdf_microfacet_fresnel_color(bsdf);
  }

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_reflect_clearcoat(const ClearcoatBsdf &bsdf,
                                                                const float3 &N,
                                                                const float3 &I,
                                                                const float3 &omega_in,
                                                                float &pdf,
                                                                const float alpha,
                                                                const float cosNO,
                                                                const float cosNI) const {
    if (!(cosNI > 0 && cosNO > 0)) {
      pdf = 0.0f;
      return make_float3(0);
    }

    /* get half vector */
    float3 m = device::normalize(omega_in + I);
    float alpha2 = alpha * alpha;
    float D, G1o, G1i;

    /* isotropic
     * eq. 20: (F*G*D)/(4*in*on)
     * eq. 33: first we calculate D(m) */
    float cosThetaM = device::dot(N, m);
    float cosThetaM2 = cosThetaM * cosThetaM;
    float cosThetaM4 = cosThetaM2 * cosThetaM2;
    float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;

    /* use GTR1 for clearcoat */
    D = D_GTR1(cosThetaM, bsdf.alpha);

    /* the alpha value for clearcoat is a fixed 0.25 => alpha2 = 0.25 * 0.25 */
    alpha2 = 0.0625f;

    /* eq. 34: now calculate G1(i,m) and G1(o,m) */
    G1o = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
    G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

    float G = G1o * G1i;

    /* eq. 20 */
    float common_ = D * 0.25f / cosNO;

    float3 F = reflection_color(bsdf, omega_in, m) * 0.25f * bsdf.clearcoat;

    float3 out_ = F * G * common_;

    pdf = G1o * common_;

    return out_;
  }

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_clearcoat(const ClearcoatBsdf &bsdf,
                                                        const float3 &I,
                                                        const float3 &omega_in,
                                                        float &pdf) const {
    const float alpha = bsdf.alpha;
    const float3 N = bsdf.N;
    const float cosNO = device::dot(N, I);
    const float cosNI = device::dot(N, omega_in);

    if (cosNI < 0.0f || alpha * alpha <= 1e-7f) {
      pdf = 0.0f;
      return make_float3(0.0);
    }

    return bsdf_microfacet_ggx_eval_reflect_clearcoat(bsdf, N, I, omega_in, pdf, alpha, cosNO, cosNI);
  }

  SPARKIUM_HD int bsdf_microfacet_ggx_sample_clearcoat(const ClearcoatBsdf &bsdf,
                                                       const float3 &Ng,
                                                       const float3 &I,
                                                       float randu,
                                                       float randv,
                                                       float3 &eval,
                                                       float3 &omega_in,
                                                       float &pdf) const {
    float alpha = bsdf.alpha;

    float3 N = bsdf.N;
    int label;

    float cosNO = device::dot(N, I);
    if (cosNO > 0) {
      float3 X, Y, Z = N;

      MakeOrthonormals(Z, X, Y);

      /* importance sampling with distribution of visible normals. vectors are
       * transformed to local space before and after */
      float3 local_I = make_float3(device::dot(X, I), device::dot(Y, I), cosNO);
      float3 local_m;
      float G1o;

      local_m = microfacet_sample_stretched(local_I, alpha, alpha, randu, randv, false, G1o);

      float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
      float cosThetaM = local_m.z;

      /* reflection or refraction? */
      float cosMO = device::dot(m, I);
      label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_GLOSSY;

      if (cosMO > 0) {
        /* eq. 39 - compute actual reflected direction */
        omega_in = 2 * cosMO * m - I;

        if (device::dot(Ng, omega_in) > 0) {
          if (alpha * alpha <= 1e-7f) {
            /* some high number for MIS */
            pdf = 1e6f;
            eval = make_float3(1e6f) * reflection_color(bsdf, omega_in, m);

            label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_SINGULAR;
          } else {
            /* microfacet normal is visible to this ray */
            /* eq. 33 */
            float alpha2 = alpha * alpha;
            float D, G1i;

            /* isotropic */
            float cosThetaM2 = cosThetaM * cosThetaM;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;

            /* eval BRDF*cosNI */
            float cosNI = device::dot(N, omega_in);

            /* use GTR1 for clearcoat */
            D = D_GTR1(cosThetaM, bsdf.alpha);

            /* the alpha value for clearcoat is a fixed 0.25 => alpha2 = 0.25 * 0.25 */
            alpha2 = 0.0625f;

            /* recalculate G1o */
            G1o = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));

            /* eq. 34: now calculate G1(i,m) */
            G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

            /* see eval function for derivation */
            float common_ = (G1o * D) * 0.25f / cosNO;
            pdf = common_;

            float3 F = reflection_color(bsdf, omega_in, m);

            eval = G1i * common_ * F;
          }

          eval *= 0.25f * bsdf.clearcoat;
        } else {
          eval = make_float3(0);
          pdf = 0.0f;
        }
      }
    } else {
      label = SPARKIUM_LABEL_REFLECT | SPARKIUM_LABEL_GLOSSY;
    }
    return label;
  }

  // -------------------------------------------------------------------------
  // principled_microfacet_refraction.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_transmit_refraction(const RefractionBsdf &bsdf,
                                                                  const float3 &N,
                                                                  const float3 &I,
                                                                  const float3 &omega_in,
                                                                  float &pdf,
                                                                  const float alpha,
                                                                  const float cosNO,
                                                                  const float cosNI) const {
    if (cosNO <= 0 || cosNI >= 0) {
      pdf = 0.0f;
      return make_float3(0); /* vectors on same side -- not possible */
    }
    /* compute half-vector of the refraction (eq. 16) */
    float m_eta = bsdf.ior;
    float3 ht = -(m_eta * omega_in + I);
    float3 Ht = device::normalize(ht);
    float cosHO = device::dot(Ht, I);
    float cosHI = device::dot(Ht, omega_in);

    float D, G1o, G1i;

    /* eq. 33: first we calculate D(m) with m=Ht: */
    float alpha2 = alpha * alpha;
    float cosThetaM = device::dot(N, Ht);
    float cosThetaM2 = cosThetaM * cosThetaM;
    float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
    float cosThetaM4 = cosThetaM2 * cosThetaM2;
    D = alpha2 / (SPARKIUM_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

    /* eq. 34: now calculate G1(i,m) and G1(o,m) */
    G1o = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
    G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

    float G = G1o * G1i;

    /* probability */
    float Ht2 = device::dot(ht, ht);

    /* out = fabsf(cosHI * cosHO) * (m_eta * m_eta) * G * D / (cosNO * Ht2)
     * pdf = pm * (m_eta * m_eta) * fabsf(cosHI) / Ht2 */
    float common_ = D * (m_eta * m_eta) / (cosNO * Ht2);
    float out_ = G * fabsf(cosHI * cosHO) * common_;
    pdf = G1o * fabsf(cosHO * cosHI) * common_;

    return make_float3(out_);
  }

  SPARKIUM_HD float3 bsdf_microfacet_ggx_eval_refraction(const RefractionBsdf &bsdf,
                                                         const float3 &I,
                                                         const float3 &omega_in,
                                                         float &pdf) const {
    const float alpha = bsdf.alpha;
    const float3 N = bsdf.N;
    const float cosNO = device::dot(N, I);
    const float cosNI = device::dot(N, omega_in);

    if (!(cosNI < 0.0f) || alpha * alpha <= 1e-7f) {
      pdf = 0.0f;
      return make_float3(0.0);
    }

    return bsdf_microfacet_ggx_eval_transmit_refraction(bsdf, N, I, omega_in, pdf, alpha, cosNO, cosNI);
  }

  SPARKIUM_HD int bsdf_microfacet_ggx_sample_refraction(const RefractionBsdf &bsdf,
                                                        const float3 &Ng,
                                                        const float3 &I,
                                                        float randu,
                                                        float randv,
                                                        float3 &eval,
                                                        float3 &omega_in,
                                                        float &pdf) const {
    float alpha = bsdf.alpha;

    float3 N = bsdf.N;
    int label;

    float cosNO = device::dot(N, I);
    if (cosNO > 0) {
      float3 X, Y, Z = N;

      MakeOrthonormals(Z, X, Y);

      /* importance sampling with distribution of visible normals. vectors are
       * transformed to local space before and after */
      float3 local_I = make_float3(device::dot(X, I), device::dot(Y, I), cosNO);
      float3 local_m;
      float G1o;

      local_m = microfacet_sample_stretched(local_I, alpha, alpha, randu, randv, false, G1o);

      float3 m = X * local_m.x + Y * local_m.y + Z * local_m.z;
      float cosThetaM = local_m.z;

      label = SPARKIUM_LABEL_TRANSMIT | SPARKIUM_LABEL_GLOSSY;

      /* CAUTION: the i and o variables are inverted relative to the paper
       * eq. 39 - compute actual refractive direction */
      float3 R, T;
      float m_eta = bsdf.ior, fresnel;
      bool inside;

      fresnel = fresnel_dielectric(m_eta, m, I, R, T, inside);

      if (!inside && fresnel != 1.0f) {
        omega_in = T;
        (void)cosThetaM;

        if (alpha * alpha <= 1e-7f || fabsf(m_eta - 1.0f) < 1e-4f) {
          /* some high number for MIS */
          pdf = 1e6f;
          eval = make_float3(1e6f);
          label = SPARKIUM_LABEL_TRANSMIT | SPARKIUM_LABEL_SINGULAR;
        } else {
          /* eq. 33 */
          float alpha2 = alpha * alpha;
          float cosThetaM2 = cosThetaM * cosThetaM;
          float cosThetaM4 = cosThetaM2 * cosThetaM2;
          float tanThetaM2 = 1 / (cosThetaM2) - 1;
          float D = alpha2 / (SPARKIUM_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));

          /* eval BRDF*cosNI */
          float cosNI = device::dot(N, omega_in);

          /* eq. 34: now calculate G1(i,m) */
          float G1i = 2 / (1 + safe_sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI)));

          /* eq. 21 */
          float cosHI = device::dot(m, omega_in);
          float cosHO = device::dot(m, I);
          float Ht2 = m_eta * cosHI + cosHO;
          Ht2 *= Ht2;

          /* see eval function for derivation */
          float common_ = (G1o * D) * (m_eta * m_eta) / (cosNO * Ht2);
          float out_ = G1i * fabsf(cosHI * cosHO) * common_;
          pdf = cosHO * fabsf(cosHI) * common_;

          eval = make_float3(out_);
        }
      } else {
        eval = make_float3(0);
        pdf = 0.0f;
      }
    } else {
      label = SPARKIUM_LABEL_TRANSMIT | SPARKIUM_LABEL_GLOSSY;
    }
    return label;
  }

  // -------------------------------------------------------------------------
  // principled_bsdf.hlsli
  // -------------------------------------------------------------------------

  SPARKIUM_HD void CalculateClosureWeight() {
    diffuse_closure.weight = make_float3(0);
    diffuse_closure.sample_weight = 0.0;
    diffuse_closure.N = make_float3(0);
    diffuse_closure.roughness = 0.0;
    microfacet_closure.weight = make_float3(0);
    microfacet_closure.sample_weight = 0.0;
    microfacet_closure.N = make_float3(0);
    microfacet_closure.alpha_x = 0.0;
    microfacet_closure.alpha_y = 0.0;
    microfacet_closure.ior = 1.0;
    microfacet_closure.T = make_float3(0);
    microfacet_closure.color = make_float3(0);
    microfacet_closure.cspec0 = make_float3(0);
    microfacet_closure.fresnel_color = make_float3(0);

    microfacet_bsdf_reflect_closure.weight = make_float3(0);
    microfacet_bsdf_reflect_closure.sample_weight = 0.0;
    microfacet_bsdf_reflect_closure.N = make_float3(0);
    microfacet_bsdf_reflect_closure.alpha_x = 0.0;
    microfacet_bsdf_reflect_closure.alpha_y = 0.0;
    microfacet_bsdf_reflect_closure.ior = 1.0;
    microfacet_bsdf_reflect_closure.T = make_float3(0);
    microfacet_bsdf_reflect_closure.color = make_float3(0);
    microfacet_bsdf_reflect_closure.cspec0 = make_float3(0);
    microfacet_bsdf_reflect_closure.fresnel_color = make_float3(0);

    microfacet_bsdf_refract_closure.weight = make_float3(0);
    microfacet_bsdf_refract_closure.sample_weight = 0.0;
    microfacet_bsdf_refract_closure.N = make_float3(0);
    microfacet_bsdf_refract_closure.alpha = 0.0;
    microfacet_bsdf_refract_closure.ior = 1.0;

    microfacet_clearcoat_closure.weight = make_float3(0);
    microfacet_clearcoat_closure.sample_weight = 0.0;
    microfacet_clearcoat_closure.N = make_float3(0);
    microfacet_clearcoat_closure.alpha = 0.0;
    microfacet_clearcoat_closure.ior = 1.0;
    microfacet_clearcoat_closure.cspec0 = make_float3(0);
    microfacet_clearcoat_closure.fresnel_color = make_float3(0);
    microfacet_clearcoat_closure.clearcoat = 0.0;

    sheen_closure.weight = make_float3(0);
    sheen_closure.sample_weight = 0.0;
    sheen_closure.N = make_float3(0);
    sheen_closure.avg_value = 0.0;

    const float3 Ng = hit_record.geom_normal;
    const float3 N = hit_record.normal;
    const float3 V = omega_v;
    const float3 I = omega_v;
    (void)Ng;
    (void)V;
    float3 T = hit_record.tangent;
    if (anisotropic_rotation != 0.0f)
      T = rotate_around_axis(T, N, anisotropic_rotation * 2.0 * SPARKIUM_PI);
    // CalculateClosureWeight is called once for direct evaluation and again for
    // path sampling.  Keep the material parameter immutable so back-face calls
    // do not alternate between eta and 1/eta.
    const float interface_ior = hit_record.front_facing ? ior : 1.0f / device::max(ior, 1e-6f);

    // calculate fresnel for refraction
    float cosNO = device::dot(N, I);
    float fresnel = fresnel_dielectric_cos(cosNO, interface_ior);

    // calculate weights of the diffuse and specular part
    float diffuse_weight =
        (1.0f - saturatef(metallic)) * (1.0f - saturatef(transmission)) * (1.0f - saturatef(subsurface));

    float final_transmission = saturatef(transmission) * (1.0f - saturatef(metallic));
    float specular_weight = (1.0f - final_transmission);
    float3 clearcoat_normal = N;
    float3 weight = make_float3(1.0);

    if (diffuse_weight > SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      float3 diff_weight = weight * base_color * diffuse_weight;

      SPARKIUM_PREPARE_BSDF(diffuse_closure, diff_weight);

      if (diffuse_closure.sample_weight > 0.0) {
        diffuse_closure.N = N;
        diffuse_closure.roughness = roughness;
      }
    }

    if (diffuse_weight > SPARKIUM_CLOSURE_WEIGHT_CUTOFF && sheen > SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      float m_cdlum = device::dot(make_float3(0.2126729f, 0.7151522f, 0.0721750f), base_color);
      float3 m_ctint = m_cdlum > 0.0f ? base_color / m_cdlum : make_float3(1);

      /* color of the sheen component */
      float3 sheen_color = make_float3(1.0f - sheen_tint) + m_ctint * sheen_tint;

      float3 sheen_weight = weight * sheen * sheen_color * diffuse_weight;

      SPARKIUM_PREPARE_BSDF(sheen_closure, sheen_weight);

      {
        sheen_closure.N = N;
        bsdf_principled_sheen_setup(sheen_closure);
      }
    }

    if (specular_weight > SPARKIUM_CLOSURE_WEIGHT_CUTOFF &&
        (specular > SPARKIUM_CLOSURE_WEIGHT_CUTOFF || metallic > SPARKIUM_CLOSURE_WEIGHT_CUTOFF)) {
      float3 spec_weight = weight * specular_weight;

      SPARKIUM_PREPARE_BSDF(microfacet_closure, spec_weight);

      {
        microfacet_closure.N = N;
        microfacet_closure.ior = (2.0f / (1.0f - safe_sqrtf(0.08f * specular))) - 1.0f;
        microfacet_closure.T = T;

        float aspect = safe_sqrtf(1.0f - anisotropic * 0.9f);
        float r2 = roughness * roughness;

        microfacet_closure.alpha_x = r2 / aspect;
        microfacet_closure.alpha_y = r2 * aspect;

        float m_cdlum = 0.3f * base_color.x + 0.6f * base_color.y + 0.1f * base_color.z;
        float3 m_ctint = m_cdlum > 0.0f ? base_color / m_cdlum : make_float3(1);
        float3 tmp_col = make_float3(1.0f - specular_tint) + m_ctint * specular_tint;

        microfacet_closure.cspec0 = ((specular * 0.08f * tmp_col) * (1.0f - metallic) + base_color * metallic);
        microfacet_closure.color = (base_color);

        bsdf_microfacet_ggx_fresnel_setup(microfacet_closure);
      }
    }

    if (final_transmission > SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      float3 glass_weight = weight * final_transmission;
      float3 cspec0 = base_color * specular_tint + make_float3(1.0f - specular_tint);
      float refl_roughness = roughness;

      /* reflection */
      {
        // This renderer represents transmission as separate reflection and
        // refraction closures.  Weight the reflection closure by the interface
        // Fresnel term before its microfacet response, matching the coefficient
        // used by the original Principled implementation.
        SPARKIUM_PREPARE_BSDF(microfacet_bsdf_reflect_closure, glass_weight * fresnel);

        {
          microfacet_bsdf_reflect_closure.N = N;
          microfacet_bsdf_reflect_closure.T = make_float3(0);

          microfacet_bsdf_reflect_closure.alpha_x = refl_roughness * refl_roughness;
          microfacet_bsdf_reflect_closure.alpha_y = refl_roughness * refl_roughness;
          microfacet_bsdf_reflect_closure.ior = interface_ior;

          microfacet_bsdf_reflect_closure.color = base_color;
          microfacet_bsdf_reflect_closure.cspec0 = cspec0;

          /* setup bsdf */
          bsdf_microfacet_ggx_fresnel_setup(microfacet_bsdf_reflect_closure);
        }
      }

      /* refraction */
      {
        /* This is to prevent MNEE from receiving a null BSDF. */
        float refraction_fresnel = device::max(0.0001f, 1.0f - fresnel);
        SPARKIUM_PREPARE_BSDF(microfacet_bsdf_refract_closure, base_color * glass_weight * refraction_fresnel);
        {
          microfacet_bsdf_refract_closure.N = N;

          const float refract_roughness = 1.0f - (1.0f - refl_roughness) * (1.0f - transmission_roughness);
          microfacet_bsdf_refract_closure.alpha = saturatef(refract_roughness * refract_roughness);
          microfacet_bsdf_refract_closure.ior = interface_ior;
        }
      }
    }

    if (clearcoat > SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      SPARKIUM_PREPARE_BSDF(microfacet_clearcoat_closure, weight);

      {
        microfacet_clearcoat_closure.N = clearcoat_normal;
        microfacet_clearcoat_closure.ior = 1.5f;
        microfacet_clearcoat_closure.alpha = clearcoat_roughness * clearcoat_roughness;
        microfacet_clearcoat_closure.cspec0 = make_float3(0.04f);
        microfacet_clearcoat_closure.clearcoat = clearcoat;

        /* setup bsdf */
        bsdf_microfacet_ggx_clearcoat_setup(microfacet_clearcoat_closure);
      }
    }
  }

  SPARKIUM_HD float3 EvalPrincipledBSDFKernel(const float3 &omega_in,
                                              float &pdf,
                                              float3 eval,
                                              float accum_weight,
                                              int exclude) const {
    float local_pdf;
    if (exclude != 0 && diffuse_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_principled_diffuse_eval(diffuse_closure, omega_v, omega_in, local_pdf) * diffuse_closure.weight;
      pdf += local_pdf * diffuse_closure.sample_weight;
      accum_weight += diffuse_closure.sample_weight;
    }
    if (exclude != 1 && microfacet_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_microfacet_ggx_eval_fresnel(microfacet_closure, omega_v, omega_in, local_pdf) *
              microfacet_closure.weight;
      pdf += local_pdf * microfacet_closure.sample_weight;
      accum_weight += microfacet_closure.sample_weight;
    }
    if (exclude != 2 && microfacet_bsdf_reflect_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_microfacet_ggx_eval_fresnel(microfacet_bsdf_reflect_closure, omega_v, omega_in, local_pdf) *
              microfacet_bsdf_reflect_closure.weight;
      pdf += local_pdf * microfacet_bsdf_reflect_closure.sample_weight;
      accum_weight += microfacet_bsdf_reflect_closure.sample_weight;
    }

    if (exclude != 3 && microfacet_bsdf_refract_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_microfacet_ggx_eval_refraction(microfacet_bsdf_refract_closure, omega_v, omega_in, local_pdf) *
              microfacet_bsdf_refract_closure.weight;
      pdf += local_pdf * microfacet_bsdf_refract_closure.sample_weight;
      accum_weight += microfacet_bsdf_refract_closure.sample_weight;
    }
    if (exclude != 4 && microfacet_clearcoat_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_microfacet_ggx_eval_clearcoat(microfacet_clearcoat_closure, omega_v, omega_in, local_pdf) *
              microfacet_clearcoat_closure.weight;
      pdf += local_pdf * microfacet_clearcoat_closure.sample_weight;
      accum_weight += microfacet_clearcoat_closure.sample_weight;
    }
    if (exclude != 5 && sheen_closure.sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_principled_sheen_eval(sheen_closure, omega_v, omega_in, local_pdf) * sheen_closure.weight;
      pdf += local_pdf * sheen_closure.sample_weight;
      accum_weight += sheen_closure.sample_weight;
    }
    if (accum_weight < SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {
      pdf = 0.0;
      return eval;
    }
    pdf /= accum_weight;
    return eval;
  }

  SPARKIUM_HD float3 EvalPrincipledBSDF(const float3 &omega_in, float &pdf) {
    CalculateClosureWeight();
    pdf = 0.0;
    return EvalPrincipledBSDFKernel(omega_in, pdf, make_float3(0.0), 0.0, -1);
  }

  SPARKIUM_HD void SamplePrincipledBSDF(float r1,
                                        float r2,
                                        float3 &eval,
                                        float3 &omega_in,
                                        float &pdf) {
    eval = make_float3(0);
    omega_in = make_float3(0);
    pdf = 0.0;
    const float3 Ng = hit_record.geom_normal;
    const float3 N = hit_record.normal;
    const float3 I = omega_v;

    CalculateClosureWeight();
    float weight_cdf[CLOSURE_COUNT];
    float total_cdf;
    weight_cdf[0] = diffuse_closure.sample_weight;
    weight_cdf[1] = microfacet_closure.sample_weight + weight_cdf[0];
    weight_cdf[2] = microfacet_bsdf_reflect_closure.sample_weight + weight_cdf[1];
    weight_cdf[3] = microfacet_bsdf_refract_closure.sample_weight + weight_cdf[2];
    weight_cdf[4] = microfacet_clearcoat_closure.sample_weight + weight_cdf[3];
    weight_cdf[5] = sheen_closure.sample_weight + weight_cdf[4];
    total_cdf = weight_cdf[CLOSURE_COUNT - 1];
    for (int i = 0; i < CLOSURE_COUNT; i++) {
      weight_cdf[i] /= total_cdf;
    }
    int exclude = -1;
    float accum_weight = 0.0;
    if (r1 < weight_cdf[0]) {
      r1 /= weight_cdf[0];
      bsdf_principled_diffuse_sample(diffuse_closure, Ng, I, r1, r2, eval, omega_in, pdf);
      eval *= diffuse_closure.weight;
      exclude = 0;
      accum_weight = diffuse_closure.sample_weight;
    } else if (r1 < weight_cdf[1]) {
      r1 -= weight_cdf[0];
      r1 /= weight_cdf[1] - weight_cdf[0];
      bsdf_microfacet_ggx_sample_fresnel(microfacet_closure, N, I, r1, r2, eval, omega_in, pdf);
      eval *= microfacet_closure.weight;

      exclude = 1;
      accum_weight = microfacet_closure.sample_weight;
    } else if (r1 < weight_cdf[2]) {
      r1 -= weight_cdf[1];
      r1 /= weight_cdf[2] - weight_cdf[1];
      bsdf_microfacet_ggx_sample_fresnel(microfacet_bsdf_reflect_closure, N, I, r1, r2, eval, omega_in, pdf);
      eval *= microfacet_bsdf_reflect_closure.weight;
      exclude = 2;
      accum_weight = microfacet_bsdf_reflect_closure.sample_weight;
    } else if (r1 < weight_cdf[3]) {
      r1 -= weight_cdf[2];
      r1 /= weight_cdf[3] - weight_cdf[2];
      bsdf_microfacet_ggx_sample_refraction(microfacet_bsdf_refract_closure, N, I, r1, r2, eval, omega_in, pdf);
      eval *= microfacet_bsdf_refract_closure.weight;
      exclude = 3;
      accum_weight = microfacet_bsdf_refract_closure.sample_weight;
    } else if (r1 < weight_cdf[4]) {
      r1 -= weight_cdf[3];
      r1 /= weight_cdf[4] - weight_cdf[3];
      bsdf_microfacet_ggx_sample_clearcoat(microfacet_clearcoat_closure, N, I, r1, r2, eval, omega_in, pdf);
      eval *= microfacet_clearcoat_closure.weight;
      exclude = 4;
      accum_weight = microfacet_clearcoat_closure.sample_weight;
    } else if (r1 < weight_cdf[5]) {
      r1 -= weight_cdf[4];
      r1 /= weight_cdf[5] - weight_cdf[4];
      bsdf_principled_sheen_sample(sheen_closure, N, I, r1, r2, eval, omega_in, pdf);
      eval *= sheen_closure.weight;
      exclude = 5;
      accum_weight = sheen_closure.sample_weight;
    }
    pdf *= accum_weight;
    eval = EvalPrincipledBSDFKernel(omega_in, pdf, eval, accum_weight, exclude);
  }
};

}  // namespace sparkium::backends
