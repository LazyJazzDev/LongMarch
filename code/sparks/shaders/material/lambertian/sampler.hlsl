#include "bindings.hlsli"
#include "direct_lighting.hlsli"

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
    context.radiance += mis_weight * eval * bsdf_eval * context.throughput;
  }

  if (max(emission.x, max(emission.y, emission.z)) > 0.0f) {
    float mis_weight = 1.0;

    if (instance_meta.custom_index != -1) {
      LightMetadata light_meta = light_metadatas[instance_meta.custom_index];
      ByteAddressBuffer direct_lighting_sampler_data = data_buffers[light_meta.sampler_data_index];
      uint primitive_index = hit_record.primitive_index;
      uint primitive_count = direct_lighting_sampler_data.Load(48);
      BufferReference<ByteAddressBuffer> power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
      float total_power = asfloat(power_cdf.Load(primitive_count * 4 - 4));
      float high_prob = asfloat(power_cdf.Load(primitive_index * 4)) / total_power;
      float low_prob = (primitive_index > 0) ? asfloat(power_cdf.Load((primitive_index - 1) * 4)) / total_power : 0.0f;
      float pdf = (high_prob - low_prob) * hit_record.pdf;
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
  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
}

#define SAMPLE_SHADOW_NO_HITRECORD

void SampleShadow(inout ShadowRayPayload payload) {
  payload.shadow = 0.0f;
}
