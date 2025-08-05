#include "bindings.hlsli"
#include "buffer_helper.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"
#include "material/principled/principled_util.hlsli"
#include "random.hlsli"

#define CLOSURE_COUNT 6

class PrincipledMaterial {
#include "material/principled/principled_bsdf.hlsli"
#include "material/principled/principled_diffuse.hlsli"
#include "material/principled/principled_microfacet.hlsli"
#include "material/principled/principled_microfacet_clearcoat.hlsli"
#include "material/principled/principled_microfacet_fresnel.hlsli"
#include "material/principled/principled_microfacet_refraction.hlsli"
#include "material/principled/principled_sheen.hlsli"

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
};

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
  if (pdf < 1e-5) {
    context.throughput = float3(0, 0, 0);
  } else {
    context.throughput *= eval / pdf;
    context.origin = hit_record.position;
    context.direction = omega_in;
    context.bsdf_pdf = pdf;
  }
  // SampleLambertianBSDF(material.base_color, context.rd, hit_record, eval, omega_in, pdf);
}

#define SAMPLE_SHADOW_NO_HITRECORD

void SampleShadow(inout ShadowRayPayload payload) {
  payload.shadow = 0.0f;
}
