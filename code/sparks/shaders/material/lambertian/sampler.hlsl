#pragma once
#include "bindings.hlsli"
#include "bsdf/lambertian.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"

void SampleMaterial(inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas[hit_record.object_index];
  ByteAddressBuffer material_buffer = data_buffers[instance_meta.material_data_index];
  float3 color = LoadFloat3(material_buffer, 0);
  float3 emission = LoadFloat3(material_buffer, 12);

  float3 eval;
  float3 omega_in;
  float pdf;
  {
    SampleDirectLighting(context, hit_record, eval, omega_in, pdf);
    float bsdf_pdf;
    float3 bsdf_eval = EvalLambertianBSDF(color, hit_record.normal, omega_in, bsdf_pdf);
    float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    eval /= pdf;
    context.shadow_eval = mis_weight * eval * bsdf_eval * context.throughput;
  }

  if (max(emission.x, max(emission.y, emission.z)) > 0.0f) {
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
  }

  SampleLambertianBSDF(color, context.rd, hit_record, eval, omega_in, pdf);
  if (pdf < EPSILON) {
    context.throughput = float3(0, 0, 0);
    return;
  }
  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
}

#define SAMPLE_SHADOW_NO_HITRECORD

void SampleShadow(inout ShadowRayPayload payload) {
  payload.shadow = 0.0f;
}
