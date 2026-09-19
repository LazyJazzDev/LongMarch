#include "native_contract.hlsli"
#pragma once

float ShadowRayNoAlpha(SP_CONTEXT float3 origin, float3 direction, float dist) {
  SP_RAY ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0f);
  ray.TMax = dist;
  SoftwareHit hit;
  return InlineIntersect(SP_CONTEXT_ARG ray, true, hit) ? 0.0f : 1.0f;
}

float ShadowRay(SP_CONTEXT float3 origin, float3 direction, float dist) {
  SP_RAY ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0f);
  ray.TMax = dist;
  float transmission = 1.0f;
  SoftwareHit hit;
  while (transmission > 1.0e-4f && InlineIntersect(SP_CONTEXT_ARG ray, false, hit)) {
    uint material = LoadSoftwareInstance(SP_BINDING_software_instances, hit.instance).material;
    transmission *= SoftwareShadowTransmission(SP_CONTEXT_ARG material,
                                               SoftwareHitRecord(SP_CONTEXT_ARG hit, direction), direction);
    // Advance one representable positive ray parameter, preserving close transparent layers.
    ray.TMin = asfloat(asuint(hit.distance) + 1);
  }
  return transmission;
}
