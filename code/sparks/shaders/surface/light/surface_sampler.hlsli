template <class BufferType>
class SurfaceSampler {
  BufferType surface_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float3 emission = LoadFloat3(surface_data, 0); // Assuming power is stored at offset 4
    uint two_sided = surface_data.Load(12); // Assuming two_sided is stored at offset 12
    float result = max(max(emission.x, emission.y), emission.z) * area * PI; // Use max to get the maximum power
    if (two_sided) {
      result *= 2.0f; // If two-sided, double the power
    }
    return result;
  }
};
