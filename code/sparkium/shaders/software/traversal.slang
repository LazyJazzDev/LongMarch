#pragma once
#include "software/layout.slang"
#ifndef SOFTWARE_EXTERNAL_BINDINGS
#include "bindings.slang"
#endif

struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint instance;
  uint primitive;
};

bool SoftwareBoxHit(SoftwareNode node, float3 origin, float3 direction, float t_min, float t_max) {
  if (any(node.lo > node.hi))
    return false;
  for (uint axis = 0; axis < 3; ++axis) {
    if (direction[axis] == 0.0f) {
      if (origin[axis] < node.lo[axis] || origin[axis] > node.hi[axis])
        return false;
    } else {
      float a = (node.lo[axis] - origin[axis]) / direction[axis];
      float b = (node.hi[axis] - origin[axis]) / direction[axis];
      t_min = max(t_min, min(a, b));
      t_max = min(t_max, max(a, b));
      if (t_min > t_max)
        return false;
    }
  }
  return true;
}

bool SoftwareTriangleHit(ByteAddressBuffer geometry,
                         uint primitive,
                         float3 origin,
                         float3 direction,
                         float t_min,
                         inout SoftwareHit hit) {
  uint3 ids = geometry.Load3(geometry.Load(48) + primitive * 12);
  uint offset = geometry.Load(8), stride = geometry.Load(12);
  float3 a = LoadFloat3(geometry, offset + stride * ids.x);
  float3 b = LoadFloat3(geometry, offset + stride * ids.y);
  float3 c = LoadFloat3(geometry, offset + stride * ids.z);
  // Watertight shear/permutation test; shared edges use the same arithmetic.
  float3 abs_dir = abs(direction);
  uint kz = abs_dir.x > abs_dir.y ? 0 : 1;
  if (abs_dir.z > abs_dir[kz])
    kz = 2;
  if (direction[kz] == 0.0f)
    return false;
  uint kx = (kz + 1) % 3, ky = (kx + 1) % 3;
  if (direction[kz] < 0.0f) {
    uint swap_axis = kx;
    kx = ky;
    ky = swap_axis;
  }

  float sx = direction[kx] / direction[kz], sy = direction[ky] / direction[kz];
  float sz = 1.0f / direction[kz];
  a -= origin;
  b -= origin;
  c -= origin;
  precise float ax = a[kx] - sx * a[kz], ay = a[ky] - sy * a[kz];
  precise float bx = b[kx] - sx * b[kz], by = b[ky] - sy * b[kz];
  precise float cx = c[kx] - sx * c[kz], cy = c[ky] - sy * c[kz];
  precise float u = cx * by - cy * bx;
  precise float v = ax * cy - ay * cx;
  precise float w = bx * ay - by * ax;
  if ((min(u, min(v, w)) < 0.0f) && (max(u, max(v, w)) > 0.0f))
    return false;
  float determinant = u + v + w;
  if (determinant == 0.0f)
    return false;
  float distance = (u * a[kz] + v * b[kz] + w * c[kz]) * sz / determinant;
  if (!(distance >= t_min && distance < hit.distance))
    return false;
  hit.distance = distance;
  hit.barycentric = float2(v, w) / determinant;
  hit.primitive = primitive;
  return true;
}

bool SoftwareTraceMesh(SoftwareInstance instance,
                       uint instance_index,
                       RayDesc ray,
                       bool any_hit,
                       inout SoftwareHit hit) {
  float3 origin = mul(instance.world_to_object, float4(ray.Origin, 1));
  // Do not normalize: the parameter t must remain in world-ray units under scaling.
  float3 direction = mul(instance.world_to_object, float4(ray.Direction, 0));
  ByteAddressBuffer geometry = data_buffers[NonUniformResourceIndex(instance.geometry)];
  uint stack[32], size = 1;
  stack[0] = instance.root;
  bool found = false;
  while (size != 0) {
    SoftwareNode node = LoadSoftwareNode(software_nodes, stack[--size]);
    if (!SoftwareBoxHit(node, origin, direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SOFTWARE_INVALID) {
      if (node.second != SOFTWARE_INVALID &&
          SoftwareTriangleHit(geometry, node.second, origin, direction, ray.TMin, hit)) {
        found = true;
        hit.instance = instance_index;
        if (any_hit)
          return true;
      }
    } else {
      stack[size++] = node.second;
      stack[size++] = node.first;
    }
  }
  return found;
}

bool InlineIntersect(RayDesc ray, bool any_hit, out SoftwareHit hit) {
  hit = (SoftwareHit)0;
  hit.distance = ray.TMax;
  hit.instance = hit.primitive = SOFTWARE_INVALID;
  if (software_instances.Load(0) == 0)
    return false;
  uint stack[32], size = 1;
  stack[0] = 0;
  bool found = false;
  while (size != 0) {
    SoftwareNode node = LoadSoftwareNode(software_nodes, stack[--size]);
    if (!SoftwareBoxHit(node, ray.Origin, ray.Direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SOFTWARE_INVALID) {
      if (node.second != SOFTWARE_INVALID &&
          SoftwareTraceMesh(LoadSoftwareInstance(software_instances, node.second), node.second, ray, any_hit, hit)) {
        found = true;
        if (any_hit)
          return true;
      }
    } else {
      stack[size++] = node.second;
      stack[size++] = node.first;
    }
  }
  return found;
}
