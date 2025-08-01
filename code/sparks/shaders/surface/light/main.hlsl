#include "bindings.hlsli"

[shader("callable")] void CallableMain(inout RenderContext context) {
  InstanceMetadata instance_meta = instance_metadatas[context.hit_record.object_id];
  ByteAddressBuffer surface_buffer = data_buffers[instance_meta.surface_data_index];
  float3 emission = LoadFloat3(surface_buffer, 0);
  int two_sided = surface_buffer.Load(12);
  int block_ray = surface_buffer.Load(16);
  if (two_sided || context.hit_record.front_facing) {
    context.radiance += emission * context.throughput;
  }
  if (block_ray) {
    context.throughput = float3(0.0, 0.0, 0.0);
  }
  context.origin = context.hit_record.position;
}
