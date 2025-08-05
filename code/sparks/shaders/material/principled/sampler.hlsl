#include "bindings.hlsli"
#include "buffer_helper.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"
#include "material/principled/principled_diffuse.hlsli"

#ifndef Spectrum
#define Spectrum float3
#endif

#define CLOSURE_COUNT 1

class PrincipledMaterial {
  HitRecord hit_record;
  float3 omega_v;
  PrincipledDiffuseBsdf diffuse_closure;
  // FresnelBsdf microfacet_closure;
  // FresnelBsdf microfacet_bsdf_reflect_closure;
  // RefractionBsdf microfacet_bsdf_refract_closure;
  // ClearcoatBsdf microfacet_clearcoat_closure;
  // PrincipledSheenBsdf sheen_closure;

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

  void CalculateClosureWeight() {
    diffuse_closure.weight = float3(0, 0, 0);
    diffuse_closure.sample_weight = 0.0;
    diffuse_closure.N = float3(0, 0, 0);
    diffuse_closure.roughness = 0.0;
    /*microfacet_closure.weight = float3(0);
    microfacet_closure.sample_weight = 0.0;
    microfacet_closure.N = float3(0);
    microfacet_closure.alpha_x = 0.0;
    microfacet_closure.alpha_y = 0.0;
    microfacet_closure.ior = 1.0;
    microfacet_closure.T = float3(0);
    microfacet_closure.color = float3(0);
    microfacet_closure.cspec0 = float3(0);
    microfacet_closure.fresnel_color = float3(0);

    microfacet_bsdf_reflect_closure.weight = float3(0);
    microfacet_bsdf_reflect_closure.sample_weight = 0.0;
    microfacet_bsdf_reflect_closure.N = float3(0);
    microfacet_bsdf_reflect_closure.alpha_x = 0.0;
    microfacet_bsdf_reflect_closure.alpha_y = 0.0;
    microfacet_bsdf_reflect_closure.ior = 1.0;
    microfacet_bsdf_reflect_closure.T = float3(0);
    microfacet_bsdf_reflect_closure.color = float3(0);
    microfacet_bsdf_reflect_closure.cspec0 = float3(0);
    microfacet_bsdf_reflect_closure.fresnel_color = float3(0);

    microfacet_bsdf_refract_closure.weight = float3(0);
    microfacet_bsdf_refract_closure.sample_weight = 0.0;
    microfacet_bsdf_refract_closure.N = float3(0);
    microfacet_bsdf_refract_closure.alpha = 0.0;
    microfacet_bsdf_refract_closure.ior = 1.0;

    microfacet_clearcoat_closure.weight = float3(0);
    microfacet_clearcoat_closure.sample_weight = 0.0;
    microfacet_clearcoat_closure.N = float3(0);
    microfacet_clearcoat_closure.alpha = 0.0;
    microfacet_clearcoat_closure.ior = 1.0;
    microfacet_clearcoat_closure.cspec0 = float3(0);
    microfacet_clearcoat_closure.fresnel_color = float3(0);
    microfacet_clearcoat_closure.clearcoat = 0.0;

    sheen_closure.weight = float3(0);
    sheen_closure.sample_weight = 0.0;
    sheen_closure.N = float3(0);
    sheen_closure.avg_value = 0.0; //*/

    const float3 Ng = hit_record.geom_normal;
    const float3 N = hit_record.normal;
    const float3 V = omega_v;
    const float3 I = omega_v;
    float3 T = hit_record.tangent;
    if (anisotropic_rotation != 0.0f)
      T = rotate_around_axis(T, N, anisotropic_rotation * 2.0 * PI);
    ior = hit_record.front_facing ? ior : 1.0f / ior;

    // calculate fresnel for refraction
    float cosNO = dot(N, I);
    float fresnel = fresnel_dielectric_cos(cosNO, ior);

    // calculate weights of the diffuse and specular part
    float diffuse_weight = (1.0f - saturatef(metallic)) * (1.0f - saturatef(transmission));

    /*float final_transmission =
        saturatef(transmission) * (1.0f - saturatef(metallic));
    float specular_weight = (1.0f - final_transmission);
    float3 clearcoat_normal = N; //*/
    Spectrum weight = float3(1.0, 1.0, 1.0);

    if (diffuse_weight > CLOSURE_WEIGHT_CUTOFF) {
      Spectrum diff_weight = weight * base_color * diffuse_weight;

      PREPARE_BSDF(diffuse_closure, diff_weight);

      if (diffuse_closure.sample_weight > 0.0) {
        diffuse_closure.N = N;
        diffuse_closure.roughness = roughness;
      }
    }
    /*
    if (diffuse_weight > CLOSURE_WEIGHT_CUTOFF && sheen > CLOSURE_WEIGHT_CUTOFF) {
      float m_cdlum = dot(float3(0.2126729f, 0.7151522f, 0.0721750f), base_color);
      float3 m_ctint = m_cdlum > 0.0f
                           ? base_color / m_cdlum
                           : float3(1);  // normalize lum. to isolate hue+sat

      float3 sheen_color = float3(1.0f - sheen_tint) + m_ctint * sheen_tint;

      Spectrum sheen_weight = weight * sheen * sheen_color * diffuse_weight;

      PREPARE_BSDF(sheen_closure, sheen_weight);

      {
        sheen_closure.N = N;
        bsdf_principled_sheen_setup(sheen_closure);
      }
    }

    if (specular_weight > CLOSURE_WEIGHT_CUTOFF &&
        (specular > CLOSURE_WEIGHT_CUTOFF || metallic > CLOSURE_WEIGHT_CUTOFF)) {
      Spectrum spec_weight = weight * specular_weight;

      PREPARE_BSDF(microfacet_closure, spec_weight);

      {
        microfacet_closure.N = N;
        microfacet_closure.ior =
            (2.0f / (1.0f - safe_sqrtf(0.08f * specular))) - 1.0f;
        microfacet_closure.T = T;

        float aspect = safe_sqrtf(1.0f - anisotropic * 0.9f);
        float r2 = roughness * roughness;

        microfacet_closure.alpha_x = r2 / aspect;
        microfacet_closure.alpha_y = r2 * aspect;

        float m_cdlum = 0.3f * base_color.x + 0.6f * base_color.y +
                        0.1f * base_color.z;  // luminance approx.
        float3 m_ctint = m_cdlum > 0.0f
                             ? base_color / m_cdlum
                             : float3(1.0);  // normalize lum. to isolate hue+sat
        float3 tmp_col = float3(1.0f - specular_tint) + m_ctint * specular_tint;

        microfacet_closure.cspec0 =
            ((specular * 0.08f * tmp_col) * (1.0f - metallic) +
             base_color * metallic);
        microfacet_closure.color = (base_color);

        bsdf_microfacet_ggx_fresnel_setup(microfacet_closure);
      }
    }

    if (final_transmission > CLOSURE_WEIGHT_CUTOFF) {
      Spectrum glass_weight = weight * final_transmission;
      float3 cspec0 = base_color * specular_tint + float3(1.0f - specular_tint);
      float refl_roughness = roughness;

      {
        PREPARE_BSDF(microfacet_bsdf_reflect_closure, glass_weight * fresnel);

        {
          microfacet_bsdf_reflect_closure.N = N;
          microfacet_bsdf_reflect_closure.T = float3(0);

          microfacet_bsdf_reflect_closure.alpha_x =
              refl_roughness * refl_roughness;
          microfacet_bsdf_reflect_closure.alpha_y =
              refl_roughness * refl_roughness;
          microfacet_bsdf_reflect_closure.ior = ior;

          microfacet_bsdf_reflect_closure.color = base_color;
          microfacet_bsdf_reflect_closure.cspec0 = cspec0;

          bsdf_microfacet_ggx_fresnel_setup(microfacet_bsdf_reflect_closure);
        }
      }

      {
        float refraction_fresnel = max(0.0001f, 1.0f - fresnel);
        PREPARE_BSDF(microfacet_bsdf_refract_closure,
                     base_color * glass_weight * refraction_fresnel);
        {
          microfacet_bsdf_refract_closure.N = N;

          transmission_roughness =
              1.0f - (1.0f - refl_roughness) * (1.0f - transmission_roughness);
          microfacet_bsdf_refract_closure.alpha =
              saturatef(transmission_roughness * transmission_roughness);
          microfacet_bsdf_refract_closure.ior = ior;
        }
      }
    }

    if (clearcoat > CLOSURE_WEIGHT_CUTOFF) {
      PREPARE_BSDF(microfacet_clearcoat_closure, weight);

      {
        microfacet_clearcoat_closure.N = clearcoat_normal;
        microfacet_clearcoat_closure.ior = 1.5f;
        microfacet_clearcoat_closure.alpha =
            clearcoat_roughness * clearcoat_roughness;
        microfacet_clearcoat_closure.cspec0 = float3(0.04f);
        microfacet_clearcoat_closure.clearcoat = clearcoat;

        bsdf_microfacet_ggx_clearcoat_setup(microfacet_clearcoat_closure);
      }
    }//*/
  }

  float3 EvalPrincipledBSDFKernel(in float3 omega_in,
                                  inout float pdf,
                                  in float3 eval,
                                  in float accum_weight,
                                  int exclude) {
    float local_pdf;
    if (exclude != 0 && diffuse_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
      eval += bsdf_principled_diffuse_eval(diffuse_closure, omega_v, omega_in, local_pdf) * diffuse_closure.weight;
      pdf += local_pdf * diffuse_closure.sample_weight;
      accum_weight += diffuse_closure.sample_weight;
    }
    /*
      if (exclude != 1 &&
          microfacet_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
        eval += bsdf_microfacet_ggx_eval_fresnel(
                    microfacet_closure, hit_record.omega_v, omega_in, local_pdf) *
                microfacet_closure.weight;
        pdf += local_pdf * microfacet_closure.sample_weight;
        accum_weight += microfacet_closure.sample_weight;
      }
      if (exclude != 2 &&
          microfacet_bsdf_reflect_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
        eval += bsdf_microfacet_ggx_eval_fresnel(microfacet_bsdf_reflect_closure,
                                                 hit_record.omega_v, omega_in,
                                                 local_pdf) *
                microfacet_bsdf_reflect_closure.weight;
        pdf += local_pdf * microfacet_bsdf_reflect_closure.sample_weight;
        accum_weight += microfacet_bsdf_reflect_closure.sample_weight;
      }

      if (exclude != 3 &&
          microfacet_bsdf_refract_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
        eval += bsdf_microfacet_ggx_eval_refraction(microfacet_bsdf_refract_closure,
                                                    hit_record.omega_v, omega_in,
                                                    local_pdf) *
                microfacet_bsdf_refract_closure.weight;
        pdf += local_pdf * microfacet_bsdf_refract_closure.sample_weight;
        accum_weight += microfacet_bsdf_refract_closure.sample_weight;
      }
      if (exclude != 4 &&
          microfacet_clearcoat_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
        eval += bsdf_microfacet_ggx_eval_clearcoat(microfacet_clearcoat_closure,
                                                   hit_record.omega_v, omega_in,
                                                   local_pdf) *
                microfacet_clearcoat_closure.weight;
        pdf += local_pdf * microfacet_clearcoat_closure.sample_weight;
        accum_weight += microfacet_clearcoat_closure.sample_weight;
      }
      if (exclude != 5 && sheen_closure.sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
        eval += bsdf_principled_sheen_eval(sheen_closure, hit_record.omega_v,
                                           omega_in, local_pdf) *
                sheen_closure.weight;
        pdf += local_pdf * sheen_closure.sample_weight;
        accum_weight += sheen_closure.sample_weight;
      } //*/
    if (accum_weight < CLOSURE_WEIGHT_CUTOFF) {
      pdf = 0.0;
      return eval;
    }
    pdf /= accum_weight;
    return eval;
  }

  float3 EvalPrincipledBSDF(in float3 omega_in, out float pdf) {
    CalculateClosureWeight();
    pdf = 0.0;
    return EvalPrincipledBSDFKernel(omega_in, pdf, float3(0, 0, 0), 0.0, -1);
  }

  void SamplePrincipledBSDF(float r1, float r2, out float3 eval, out float3 omega_in, out float pdf) {
    eval = float3(0, 0, 0);
    omega_in = float3(0, 0, 0);
    pdf = 0.0;
    const float3 Ng = hit_record.geom_normal;
    const float3 N = hit_record.normal;
    const float3 I = omega_v;

    CalculateClosureWeight();
    float weight_cdf[CLOSURE_COUNT];
    float total_cdf;
    weight_cdf[0] = diffuse_closure.sample_weight;
    /*weight_cdf[1] = microfacet_closure.sample_weight + weight_cdf[0];
    weight_cdf[2] = microfacet_bsdf_reflect_closure.sample_weight + weight_cdf[1];
    weight_cdf[3] = microfacet_bsdf_refract_closure.sample_weight + weight_cdf[2];
    weight_cdf[4] = microfacet_clearcoat_closure.sample_weight + weight_cdf[3];
    weight_cdf[5] = sheen_closure.sample_weight + weight_cdf[4]; //*/
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
    } /* else if (r1 < weight_cdf[1]) {
       r1 -= weight_cdf[0];
       r1 /= weight_cdf[1] - weight_cdf[0];
       bsdf_microfacet_ggx_sample_fresnel(microfacet_closure, N, I, r1, r2, eval,
                                          omega_in, pdf);
       eval *= microfacet_closure.weight;
       exclude = 1;
       accum_weight = microfacet_closure.sample_weight;
     } else if (r1 < weight_cdf[2]) {
       r1 -= weight_cdf[1];
       r1 /= weight_cdf[2] - weight_cdf[1];
       bsdf_microfacet_ggx_sample_fresnel(microfacet_bsdf_reflect_closure, N, I,
                                          r1, r2, eval, omega_in, pdf);
       eval *= microfacet_bsdf_reflect_closure.weight;
       exclude = 2;
       accum_weight = microfacet_bsdf_reflect_closure.sample_weight;
     } else if (r1 < weight_cdf[3]) {
       r1 -= weight_cdf[2];
       r1 /= weight_cdf[3] - weight_cdf[2];
       bsdf_microfacet_ggx_sample_refraction(microfacet_bsdf_refract_closure, N, I,
                                             r1, r2, eval, omega_in, pdf);
       eval *= microfacet_bsdf_refract_closure.weight;
       exclude = 3;
       accum_weight = microfacet_bsdf_refract_closure.sample_weight;
     } else if (r1 < weight_cdf[4]) {
       r1 -= weight_cdf[3];
       r1 /= weight_cdf[4] - weight_cdf[3];
       bsdf_microfacet_ggx_sample_clearcoat(microfacet_clearcoat_closure, N, I, r1,
                                            r2, eval, omega_in, pdf);
       eval *= microfacet_clearcoat_closure.weight;
       exclude = 4;
       accum_weight = microfacet_clearcoat_closure.sample_weight;
     } else if (r1 < weight_cdf[5]) {
       r1 -= weight_cdf[4];
       r1 /= weight_cdf[5] - weight_cdf[4];
       bsdf_principled_sheen_sample(sheen_closure, N, I, r1, r2, eval, omega_in,
                                    pdf);
       eval *= sheen_closure.weight;
       exclude = 5;
       accum_weight = sheen_closure.sample_weight;
     } //*/
    pdf *= accum_weight;
    eval = EvalPrincipledBSDFKernel(omega_in, pdf, eval, accum_weight, exclude);
  }
};

void SampleMaterial(inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas[hit_record.object_index];
  StreamedBufferReference<ByteAddressBuffer> material_buffer =
      MakeStreamedBufferReference(data_buffers[instance_meta.material_data_index], 0);

  PrincipledMaterial material;
  material.hit_record = hit_record;
  material.omega_v = -context.direction;
  material.base_color = material_buffer.LoadFloat3();
  material.subsurface_color = material_buffer.LoadFloat3();
  material.subsurface = material_buffer.LoadFloat();
  material.subsurface_radius = material_buffer.LoadFloat3();
  material.metallic = material_buffer.LoadFloat();
  material.specular = material_buffer.LoadFloat();
  material.specular_tint = material_buffer.LoadFloat();

  material.roughness = material_buffer.LoadFloat();
  material.anisotropic = material_buffer.LoadFloat();
  material.anisotropic_rotation = material_buffer.LoadFloat();
  material.sheen = material_buffer.LoadFloat();
  material.sheen_tint = material_buffer.LoadFloat();
  material.clearcoat = material_buffer.LoadFloat();
  material.clearcoat_roughness = material_buffer.LoadFloat();
  material.ior = material_buffer.LoadFloat();
  material.transmission = material_buffer.LoadFloat();
  material.transmission_roughness = material_buffer.LoadFloat();

  float3 eval;
  float3 omega_in;
  float pdf;

  {
    SampleDirectLighting(context, hit_record, eval, omega_in, pdf);
    float bsdf_pdf;
    // float3 EvalPrincipledBSDF(in float3 omega_in, out float pdf);
    float3 bsdf_eval = material.EvalPrincipledBSDF(
        omega_in, bsdf_pdf);  // EvalLambertianBSDF(color, hit_record.normal, omega_in, bsdf_pdf);
    float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    eval /= pdf;
    context.radiance += mis_weight * eval * bsdf_eval * context.throughput;
  }

  /*if (max(emission.x, max(emission.y, emission.z)) > 0.0f) {
    float mis_weight = 1.0;

    if (instance_meta.custom_index != -1) {
      LightMetadata light_meta = light_metadatas[instance_meta.custom_index];
      float pdf = hit_record.pdf *
                  EvaluatePrimitiveProbability(data_buffers[light_meta.sampler_data_index], hit_record.primitive_index);
      pdf *= DirectLightingProbability(instance_meta.custom_index);
      float3 omega_in = hit_record.position - context.origin;
      pdf *= dot(omega_in, omega_in);
      float NdotL = abs(dot(hit_record.geom_normal, normalize(omega_in)));
      pdf /= NdotL;
      mis_weight = PowerHeuristic(context.bsdf_pdf, pdf);
    }

    context.radiance += emission * context.throughput * mis_weight;
  } //*/

  material.SamplePrincipledBSDF(RandomFloat(context.rd), RandomFloat(context.rd), eval, omega_in, pdf);

  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
}

#define SAMPLE_SHADOW_NO_HITRECORD

void SampleShadow(inout ShadowRayPayload payload) {
  payload.shadow = 0.0f;
}
