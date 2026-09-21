#pragma once
#include "compute_contract.hlsli"

struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint instance;
  uint primitive;
};

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
