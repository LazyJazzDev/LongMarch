
template <class BufferType, class BufferType2>
float PowerSampler(BufferType material_data, BufferType2 geometry_data, float3x4 transform, uint primitive_id) {
  float area = PrimitiveArea(geometry_data, transform, primitive_id);
  float3 emission = LoadFloat3(material_data, 0); // Assuming power is stored at offset 4
  uint two_sided = material_data.Load(12); // Assuming two_sided is stored at offset 12
  float result = max(max(emission.x, emission.y), emission.z) * area * PI; // Use max to get the maximum power
  if (two_sided) {
    result *= 2.0f; // If two-sided, double the power
  }
  return result;
}
