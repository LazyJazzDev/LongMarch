#pragma once
template <class BufferType>
float3 MaterialLightEvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float3 emission = LoadFloat3(material_data, 0);
  uint two_sided = material_data.Load(12);
  float3 omega_in = normalize(primitive_sample.position - position);
  if (two_sided || dot(primitive_sample.normal, omega_in) < 0.0f) {
    // If two-sided or front-facing, return the emission
    return emission;
  }
  return float3(0.0f, 0.0f, 0.0f); // If back-facing, return zero contribution
}
