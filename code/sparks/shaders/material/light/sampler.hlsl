#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"

void SampleMaterial(inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta = instance_metadatas[hit_record.object_index];
  ByteAddressBuffer material_buffer = data_buffers[instance_meta.material_data_index];
  float3 emission = LoadFloat3(material_buffer, 0);
  int two_sided = material_buffer.Load(12);
  int block_ray = material_buffer.Load(16);
  if (two_sided || hit_record.front_facing) {
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

  if (block_ray) {
    context.throughput = float3(0.0, 0.0, 0.0);
  }
  context.origin = hit_record.position;
}
