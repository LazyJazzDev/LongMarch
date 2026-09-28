#pragma once

float ShadowRayNoAlpha(float3 origin, float3 direction, float dist, int2 skip) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0f);
  ray.TMax = dist;
  SoftwareHit hit;
  if (skip.x < 0)
    return InlineIntersect(ray, true, hit) ? 0.0f : 1.0f;
  while (InlineIntersect(ray, false, hit)) {
    if (any(int2(hit.instance, hit.primitive) != skip))
      return 0.0f;
    ray.TMin = asfloat(asuint(hit.distance) + 1);
  }
  return 1.0f;
}

float ShadowRay(float3 origin, float3 direction, float dist, int2 skip) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = T_MIN * max(length(origin), 1.0f);
  ray.TMax = dist;
  float transmission = 1.0f;
  SoftwareHit hit;
  while (transmission > 1.0e-4f && InlineIntersect(ray, false, hit)) {
    if (all(int2(hit.instance, hit.primitive) == skip)) {
      ray.TMin = asfloat(asuint(hit.distance) + 1);
      continue;
    }
    uint material = LoadSoftwareInstance(software_instances, hit.instance).material;
    transmission *= SoftwareShadowTransmission(material, SoftwareHitRecord(hit, direction), direction);
    // Advance one representable positive ray parameter, preserving close transparent layers.
    ray.TMin = asfloat(asuint(hit.distance) + 1);
  }
  return transmission;
}
