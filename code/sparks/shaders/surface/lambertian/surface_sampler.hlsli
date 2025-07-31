template <class BufferType>
class SurfaceSampler {
  BufferType surface_data;

  template <class GeometrySamplerType>
  float PrimitivePower(GeometrySamplerType geometry_sampler, uint primitive_id) {
    float area = geometry_sampler.PrimitiveArea(primitive_id);
    float3 emission = LoadFloat3(surface_data, 12);
    return max(max(emission.x, emission.y), emission.z) * area * PI * 2.0; // Use max to get the maximum power
  }
};
