#pragma once
#include "material/shader_graph/surface_sampler.hlsli"

#define SAMPLE_SHADOW_ANY_HIT

// SHADER_GRAPH_IMPLEMENTATION

void SampleMaterial(inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta =
      instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(instance_meta.material_data_index)];
  GraphSurface graph =
      EvaluateShaderGraph(hit_record, -context.direction, context.bounce, context.ray_type, false, material_data);
  SampleGraphSurface(context, hit_record, graph);
}

float SampleShadowOpacity(HitRecord hit_record, float3 ray_direction) {
  InstanceMetadata instance_meta =
      instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_data = data_buffers[NonUniformResourceIndex(instance_meta.material_data_index)];
  return GraphShadowOpacity(
      EvaluateShaderGraph(hit_record, -ray_direction, 1, RAY_TYPE_REFLECTION, true, material_data));
}

void SampleShadow(inout ShadowRayPayload payload, HitRecord hit_record) {
  // Opacity is accumulated by ShadowAnyHit. Reaching closest-hit means the
  // accumulated transmittance has become zero.
  payload.shadow = 0.0f;
}
