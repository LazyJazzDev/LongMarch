#ifndef SPARKIUM_SHADER_MATERIAL_PRINCIPLED_EVAL_DIRECT_LIGHT_HLSLI_
#define SPARKIUM_SHADER_MATERIAL_PRINCIPLED_EVAL_DIRECT_LIGHT_HLSLI_
template <class BufferType>
float3 MaterialPrincipledEvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float4 emission = LoadFloat4(material_data, 92);
  return emission.xyz * emission.w; // Scale by the emission intensity
}

#endif  // SPARKIUM_SHADER_MATERIAL_PRINCIPLED_EVAL_DIRECT_LIGHT_HLSLI_