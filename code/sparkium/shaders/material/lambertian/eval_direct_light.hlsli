#ifndef SPARKIUM_SHADER_MATERIAL_LAMBERTIAN_EVAL_DIRECT_LIGHT_HLSLI_
#define SPARKIUM_SHADER_MATERIAL_LAMBERTIAN_EVAL_DIRECT_LIGHT_HLSLI_
template <class BufferType>
float3 MaterialLambertianEvaluateDirectLighting(BufferType material_data, float3 position, GeometryPrimitiveSample primitive_sample) {
  float3 emission = LoadFloat3(material_data, 12);
  return emission;
}

#endif  // SPARKIUM_SHADER_MATERIAL_LAMBERTIAN_EVAL_DIRECT_LIGHT_HLSLI_