#pragma once
#define CLOSURE_BSDF_MICROFACET_GGX_FRESNEL_ID 0
#define CLOSURE_BSDF_MICROFACET_GGX_CLEARCOAT_ID 1
#define CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID 2
#define CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID 3

void microfacet_beckmann_sample_slopes(const float cos_theta_i,
                                       const float sin_theta_i,
                                       float randu,
                                       float randv,
                                       inout float slope_x,
                                       inout float slope_y,
                                       inout float G1i) {
  /* special case (normal incidence) */
  if (cos_theta_i >= 0.99999f) {
    const float r = sqrt(-log(randu));
    const float phi = 2.0 * PI * randv;
    slope_x = r * cos(phi);
    slope_y = r * sin(phi);
    G1i = 1.0f;
    return;
  }

  /* precomputations */
  const float tan_theta_i = sin_theta_i / cos_theta_i;
  const float inv_a = tan_theta_i;
  const float cot_theta_i = 1.0f / tan_theta_i;
  const float erf_a = fast_erff(cot_theta_i);
  const float exp_a2 = exp(-cot_theta_i * cot_theta_i);
  const float SQRT_PI_INV = 0.56418958354f;
  const float Lambda =
      0.5f * (erf_a - 1.0f) + (0.5f * SQRT_PI_INV) * (exp_a2 * inv_a);
  const float G1 = 1.0f / (1.0f + Lambda); /* masking */

  G1i = G1;
  /* Based on paper from Wenzel Jakob
   * An Improved Visible Normal Sampling Routine for the Beckmann Distribution
   *
   * http://www.mitsuba-renderer.org/~wenzel/files/visnormal.pdf
   *
   * Reformulation from OpenShadingLanguage which avoids using inverse
   * trigonometric functions.
   */

  /* Sample slope X.
   *
   * Compute a coarse approximation using the approximation:
   *   exp(-ierf(x)^2) ~= 1 - x * x
   *   solve y = 1 + b + K * (1 - b * b)
   */
  float K = tan_theta_i * SQRT_PI_INV;
  float y_approx = randu * (1.0f + erf_a + K * (1 - erf_a * erf_a));
  float y_exact = randu * (1.0f + erf_a + K * exp_a2);
  float b = K > 0 ? (0.5f - sqrt(K * (K - y_approx + 1.0f) + 0.25f)) / K
                  : y_approx - 1.0f;

  /* Perform newton step to refine toward the true root. */
  float inv_erf = fast_ierff(b);
  float value = 1.0f + b + K * exp(-inv_erf * inv_erf) - y_exact;
  /* Check if we are close enough already,
   * this also avoids NaNs as we get close to the root.
   */
  if (abs(value) > 1e-6f) {
    b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 1. */
    inv_erf = fast_ierff(b);
    value = 1.0f + b + K * exp(-inv_erf * inv_erf) - y_exact;
    b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 2. */
    /* Compute the slope from the refined value. */
    slope_x = fast_ierff(b);
  } else {
    /* We are close enough already. */
    slope_x = inv_erf;
  }
  slope_y = fast_ierff(2.0f * randv - 1.0f);
}

void microfacet_ggx_sample_slopes(const float cos_theta_i,
                                  const float sin_theta_i,
                                  float randu,
                                  float randv,
                                  inout float slope_x,
                                  inout float slope_y,
                                  inout float G1i) {
  /* special case (normal incidence) */
  if (cos_theta_i >= 0.99999f) {
    const float r = sqrt(randu / (1.0f - randu));
    const float phi = 2.0 * PI * randv;
    slope_x = r * cos(phi);
    slope_y = r * sin(phi);
    G1i = 1.0f;

    return;
  }

  /* precomputations */
  const float tan_theta_i = sin_theta_i / cos_theta_i;
  const float G1_inv =
      0.5f * (1.0f + safe_sqrtf(1.0f + tan_theta_i * tan_theta_i));

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
  slope_x =
      (A < 0.0f || slope_x_2 * tan_theta_i > 1.0f) ? slope_x_1 : slope_x_2;

  /* sample slope_y */
  float S;

  if (randv > 0.5f) {
    S = 1.0f;
    randv = 2.0f * (randv - 0.5f);
  } else {
    S = -1.0f;
    randv = 2.0f * (0.5f - randv);
  }

  const float z =
      (randv * (randv * (randv * 0.27385f - 0.73369f) + 0.46341f)) /
      (randv * (randv * (randv * 0.093073f + 0.309420f) - 1.000000f) +
       0.597999f);
  slope_y = S * z * safe_sqrtf(1.0f + (slope_x) * (slope_x));
}

float3 microfacet_sample_stretched(const float3 omega_i,
                                   const float alpha_x,
                                   const float alpha_y,
                                   const float randu,
                                   const float randv,
                                   bool beckmann,
                                   inout float G1i) {
  /* 1. stretch omega_i */
  float3 omega_i_ = make_float3(alpha_x * omega_i.x, alpha_y * omega_i.y, omega_i.z);
  omega_i_ = normalize(omega_i_);

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
    microfacet_beckmann_sample_slopes(costheta_, sintheta_, randu, randv,
                                      slope_x, slope_y, G1i);
  } else {
    microfacet_ggx_sample_slopes(costheta_, sintheta_, randu, randv, slope_x,
                                 slope_y, G1i);
  }

  /* 3. rotate */
  float tmp = cosphi_ * slope_x - sinphi_ * slope_y;
  slope_y = sinphi_ * slope_x + cosphi_ * slope_y;
  slope_x = tmp;

  /* 4. unstretch */
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;

  /* 5. compute normal */
  return normalize(make_float3(-slope_x, -slope_y, 1.0f));
}
