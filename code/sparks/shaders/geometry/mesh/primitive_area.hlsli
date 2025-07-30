float PrimitiveArea(ByteAddressBuffer geom_buffer, float3x4 transform, int primitive_id) {
  uint num_indices = geom_buffer.Load(4);
  uint position_offset = geom_buffer.Load(8);
  uint position_stride = geom_buffer.Load(12);
  uint index_offset = geom_buffer.Load(48);
  uint3 vid;
  vid = geom_buffer.Load3(index_offset + primitive_id * 3 * 4);
  float3 pos[3];
  pos[0] = mul(transform, float4(LoadFloat3(geom_buffer, position_offset + position_stride * vid[0]), 1.0));
  pos[1] = mul(transform, float4(LoadFloat3(geom_buffer, position_offset + position_stride * vid[1]), 1.0));
  pos[2] = mul(transform, float4(LoadFloat3(geom_buffer, position_offset + position_stride * vid[2]), 1.0));
  return length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f;
}
