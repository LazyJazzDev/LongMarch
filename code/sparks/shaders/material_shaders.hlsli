
// Evaluate Direct Lighting Implementation

template <class BufferType>
float3 EvaluateDirectLighting(int shader_index, BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  switch (shader_index) {
// EvaluateDirectLighting Function List
  default:
    return float3(0.0f, 0.0f, 0.0f);
  }
}

// Material Sampler Implementation

void SampleMaterial(int shader_index, inout RenderContext context, HitRecord hit_record) {
  switch (shader_index) {
// SampleMaterial Function List
  default:
    // Fallback behavior or error handling
    context.throughput = float3(0, 0, 0);
    break;
  }
}
