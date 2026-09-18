#pragma once

// Port of `software/layout.hlsli` and `software/traversal.hlsli`. The node and
// instance byte layouts are the ones `SoftwarePipeline` builds, and the
// builder in `native_bvh.cpp` reproduces the same tree, so traversal results
// agree with the GPU software path.

#include "native_geometry.h"

namespace sparkium::native {

// Explicit byte layouts shared with SoftwarePipeline. All indices are node indices.
static const uint32_t SOFTWARE_NODE_BYTES = 32;
static const uint32_t SOFTWARE_INSTANCE_BYTES = 112;
static const uint32_t SOFTWARE_INVALID = 0xffffffffu;

struct SoftwareNode {
  float3 lo;
  uint32_t first;
  float3 hi;
  uint32_t second;
};

struct SoftwareInstance {
  float3x4 object_to_world;
  float3x4 world_to_object;
  uint32_t root;
  uint32_t geometry;
  uint32_t material;
  uint32_t primitive_count;
};

struct RayDesc {
  float3 Origin;
  float TMin;
  float3 Direction;
  float TMax;
};

struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint32_t instance;
  uint32_t primitive;
};

LM_DEVICE_FUNC inline SoftwareNode LoadSoftwareNode(const ByteBuffer &nodes, uint32_t index) {
  const uint32_t offset = index * SOFTWARE_NODE_BYTES;
  SoftwareNode node;
  node.lo = LoadFloat3(nodes, offset);
  node.first = nodes.Load(offset + 12);
  node.hi = LoadFloat3(nodes, offset + 16);
  node.second = nodes.Load(offset + 28);
  return node;
}

LM_DEVICE_FUNC inline SoftwareInstance LoadSoftwareInstance(const ByteBuffer &instances, uint32_t index) {
  const uint32_t offset = 16 + index * SOFTWARE_INSTANCE_BYTES;
  SoftwareInstance result;
  result.object_to_world = LoadFloat3x4(instances, offset);
  result.world_to_object = LoadFloat3x4(instances, offset + 48);
  const uint4 info = instances.Load4(offset + 96);
  result.root = info.x;
  result.geometry = info.y;
  result.material = info.z;
  result.primitive_count = info.w;
  return result;
}

LM_DEVICE_FUNC inline bool SoftwareBoxHit(const SoftwareNode &node,
                                          const float3 &origin,
                                          const float3 &direction,
                                          float t_min,
                                          float t_max) {
  if (node.lo.x > node.hi.x || node.lo.y > node.hi.y || node.lo.z > node.hi.z)
    return false;
  for (uint32_t axis = 0; axis < 3; ++axis) {
    if (direction[axis] == 0.0f) {
      if (origin[axis] < node.lo[axis] || origin[axis] > node.hi[axis])
        return false;
    } else {
      const float a = (node.lo[axis] - origin[axis]) / direction[axis];
      const float b = (node.hi[axis] - origin[axis]) / direction[axis];
      t_min = ::fmaxf(t_min, ::fminf(a, b));
      t_max = ::fminf(t_max, ::fmaxf(a, b));
      if (t_min > t_max)
        return false;
    }
  }
  return true;
}

LM_DEVICE_FUNC inline bool SoftwareTriangleHit(const ByteBuffer &geometry,
                                               uint32_t primitive,
                                               const float3 &origin,
                                               const float3 &direction,
                                               float t_min,
                                               SoftwareHit &hit) {
  const uint3 ids = geometry.Load3(geometry.Load(48) + primitive * 12);
  const uint32_t offset = geometry.Load(8), stride = geometry.Load(12);
  float3 a = LoadFloat3(geometry, offset + stride * ids.x);
  float3 b = LoadFloat3(geometry, offset + stride * ids.y);
  float3 c = LoadFloat3(geometry, offset + stride * ids.z);
  // Watertight shear/permutation test; shared edges use the same arithmetic.
  const float3 abs_dir = glm::abs(direction);
  uint32_t kz = abs_dir.x > abs_dir.y ? 0 : 1;
  if (abs_dir.z > abs_dir[kz])
    kz = 2;
  if (direction[kz] == 0.0f)
    return false;
  uint32_t kx = (kz + 1) % 3, ky = (kx + 1) % 3;
  if (direction[kz] < 0.0f) {
    const uint32_t swap_axis = kx;
    kx = ky;
    ky = swap_axis;
  }
  const float sx = direction[kx] / direction[kz], sy = direction[ky] / direction[kz];
  const float sz = 1.0f / direction[kz];
  a -= origin;
  b -= origin;
  c -= origin;
  const float ax = a[kx] - sx * a[kz], ay = a[ky] - sy * a[kz];
  const float bx = b[kx] - sx * b[kz], by = b[ky] - sy * b[kz];
  const float cx = c[kx] - sx * c[kz], cy = c[ky] - sy * c[kz];
  const float u = cx * by - cy * bx;
  const float v = ax * cy - ay * cx;
  const float w = bx * ay - by * ax;
  if ((::fminf(u, ::fminf(v, w)) < 0.0f) && (::fmaxf(u, ::fmaxf(v, w)) > 0.0f))
    return false;
  const float determinant = u + v + w;
  if (determinant == 0.0f)
    return false;
  const float distance = (u * a[kz] + v * b[kz] + w * c[kz]) * sz / determinant;
  if (!(distance >= t_min && distance < hit.distance))
    return false;
  hit.distance = distance;
  hit.barycentric = float2{v, w} / determinant;
  hit.primitive = primitive;
  return true;
}

LM_DEVICE_FUNC inline bool SoftwareTraceMesh(const SceneView &scene,
                                             const SoftwareInstance &instance,
                                             uint32_t instance_index,
                                             const RayDesc &ray,
                                             bool any_hit,
                                             SoftwareHit &hit) {
  const float3 origin = mul(instance.world_to_object, float4{ray.Origin, 1});
  // Do not normalize: the parameter t must remain in world-ray units under scaling.
  const float3 direction = mul(instance.world_to_object, float4{ray.Direction, 0});
  const ByteBuffer geometry = scene.DataBuffer(static_cast<int>(instance.geometry));
  uint32_t stack[32], size = 1;
  stack[0] = instance.root;
  bool found = false;
  while (size != 0) {
    const SoftwareNode node = LoadSoftwareNode(scene.software_nodes, stack[--size]);
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

LM_DEVICE_FUNC inline bool InlineIntersect(const SceneView &scene, const RayDesc &ray, bool any_hit, SoftwareHit &hit) {
  hit.distance = ray.TMax;
  hit.barycentric = float2{0.0f, 0.0f};
  hit.instance = hit.primitive = SOFTWARE_INVALID;
  if (scene.software_instances.Load(0) == 0)
    return false;
  uint32_t stack[32], size = 1;
  stack[0] = 0;
  bool found = false;
  while (size != 0) {
    const SoftwareNode node = LoadSoftwareNode(scene.software_nodes, stack[--size]);
    if (!SoftwareBoxHit(node, ray.Origin, ray.Direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SOFTWARE_INVALID) {
      if (node.second != SOFTWARE_INVALID &&
          SoftwareTraceMesh(scene, LoadSoftwareInstance(scene.software_instances, node.second), node.second, ray,
                            any_hit, hit)) {
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

}  // namespace sparkium::native
