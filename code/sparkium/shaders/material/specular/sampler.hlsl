#include "compute_contract.hlsli"
#pragma once
#include "bindings.hlsli"
#include "bsdf/specular.hlsli"
#include "direct_lighting.hlsli"
#include "geometry_primitive_sampler.hlsli"

void SampleMaterial(SP_CONTEXT inout RenderContext context, HitRecord hit_record) {
  InstanceMetadata instance_meta =
      SP_BINDING_instance_metadatas.Load<InstanceMetadata>(sizeof(InstanceMetadata) * hit_record.object_index);
  ByteAddressBuffer material_buffer = SP_BINDING_data_buffers[SP_NONUNIFORM(instance_meta.material_data_index)];
  float3 color = LoadFloat3(material_buffer, 0);

  float3 eval;
  float3 omega_in;
  float pdf;

  SampleSpecularBSDF(color, context.direction, hit_record.normal, hit_record.geom_normal, eval, omega_in, pdf);
  context.throughput *= eval;
  context.origin = hit_record.position;
  context.direction = omega_in;
  context.bsdf_pdf = pdf;
  context.ray_type = RAY_TYPE_REFLECTION;
}

#define SAMPLE_SHADOW_NO_HITRECORD

void SampleShadow(inout ShadowRayPayload payload) {
  payload.shadow = 0.0f;
}
