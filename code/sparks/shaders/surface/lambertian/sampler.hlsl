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

void SampleLambertianBSDF(float3 base_color,
                          inout RenderContext context,
                          out float3 eval,
                          out float3 L,
                          out float pdf) {
  SampleCosHemisphere(context.rd, context.hit_record.normal, L, pdf);
  if (dot(context.hit_record.geom_normal, L) > 0.0) {
    eval = pdf * base_color;
  } else {
    eval = float3(0, 0, 0);
  }
}

void SurfaceSample(inout RenderContext2 context, HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas[hit_record.object_index];
  ByteAddressBuffer surface_buffer = data_buffers[instance_meta.surface_data_index];
  float3 color = LoadFloat3(surface_buffer, 0);
  float3 emission = LoadFloat3(surface_buffer, 12);

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

  context.radiance += emission * context.throughput;
  SampleLambertianBSDF(color, context.rd, hit_record, eval, omega_in, pdf);
  context.throughput *= eval / pdf;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
}

[shader("callable")] void SurfaceSampler(inout RenderContext context) {
  InstanceMetadata instance_meta = instance_metadatas[context.hit_record.object_index];
  ByteAddressBuffer surface_buffer = data_buffers[instance_meta.surface_data_index];
  float3 color = LoadFloat3(surface_buffer, 0);
  float3 emission = LoadFloat3(surface_buffer, 12);

  float3 eval;
  float3 omega_in;
  float pdf;
  {
    SampleDirectLighting(context, eval, omega_in, pdf);
    float bsdf_pdf;
    float3 bsdf_eval = EvalLambertianBSDF(color, context.hit_record.normal, omega_in, bsdf_pdf);
    float mis_weight = PowerHeuristic(pdf, bsdf_pdf);
    eval /= pdf;
    context.radiance += mis_weight * eval * bsdf_eval * context.throughput;
  }

  context.radiance += emission * context.throughput;
  SampleLambertianBSDF(color, context, eval, omega_in, pdf);
  context.throughput *= eval / pdf;
  context.origin = context.hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
}
