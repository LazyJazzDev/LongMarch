template <class BufferType>
class SurfaceEvaluator {
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

  float3 EvaluateDirectLighting(float3 position, GeometryPrimitiveSample primitive_sample) {
    float3 emission = LoadFloat3(surface_data, 0);
    uint two_sided = surface_data.Load(12);
    float3 omega_in = normalize(primitive_sample.position - position);
    if (two_sided || dot(primitive_sample.normal, omega_in) < 0.0f) {
      // If two-sided or front-facing, return the emission
      return emission;
    }
    return float3(0.0f, 0.0f, 0.0f); // If back-facing, return zero contribution
  }
};
