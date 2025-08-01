#include "bindings.hlsli"

[shader("callable")] void CallableMain(inout RenderContext context) {
  InstanceMetadata instance_meta = instance_metadatas[context.hit_record.object_id];
  ByteAddressBuffer surface_buffer = data_buffers[instance_meta.surface_data_index];
  float3 color = LoadFloat3(surface_buffer, 0);
  float3 emission = LoadFloat3(surface_buffer, 12);
  context.radiance += emission * context.throughput;
  float3 omega_in;
  float pdf;
  SampleCosHemisphere(context.rd, context.hit_record.normal, omega_in, pdf);
  float NdotL = dot(context.hit_record.normal, omega_in);
  if (NdotL > 0 && dot(context.hit_record.geom_normal, omega_in) > 0) {
    context.throughput *= color * NdotL * INV_PI / pdf;
  } else {
    context.throughput = float3(0.0, 0.0, 0.0);
  }
  context.origin = context.hit_record.position;
  context.direction = omega_in;
}
