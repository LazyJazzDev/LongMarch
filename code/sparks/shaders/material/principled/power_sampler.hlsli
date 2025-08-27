
template <class BufferType, class BufferType2>
float PowerSampler(BufferType material_data, BufferType2 geometry_data, float3x4 transform, uint primitive_id) {
  float area = PrimitiveArea(geometry_data, transform, primitive_id);
  float4 emission = LoadFloat4(material_data, 92);
  return max(max(emission.x, emission.y), emission.z) * emission.w * area * PI * 2.0; // Use max to get the maximum power
}
