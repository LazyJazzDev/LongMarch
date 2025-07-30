#include "bindings.hlsli"

[shader("callable")] void CallableMain(inout RenderContext context) {
  ByteAddressBuffer material_buffer = material_data[context.material_buffer_index];
  float3 emission = LoadFloat3(material_buffer, 0);
  int two_sided = material_buffer.Load(12);
  int block_ray = material_buffer.Load(16);
  if (two_sided || context.hit_record.front_facing) {
    context.radiance += emission * context.throughput;
  }
  if (block_ray) {
    context.throughput = float3(0.0, 0.0, 0.0);
  }
  context.origin = context.hit_record.position;
}
