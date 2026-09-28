#pragma once

// Project the hit toward each vertex tangent plane, but never beneath it.
// Barycentric interpolation gives a continuous, scale-aware offset on smooth
// meshes and exactly zero offset on flat faces. Use only for visibility rays;
// BSDF evaluation, textures and light PDFs retain the actual triangle hit.
float3 ShadowTerminatorOffset(float3 position, float3 vertex, float3 normal) {
  return -min(dot(position - vertex, normal), 0.0f) * normal;
}
