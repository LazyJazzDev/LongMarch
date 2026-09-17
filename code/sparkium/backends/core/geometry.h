#pragma once
// Portable port of
//   code/sparkium/shaders/software/traversal.hlsli
//   code/sparkium/shaders/geometry/mesh/hit_record.hlsli
//   code/sparkium/shaders/geometry/mesh/sample_primitive.hlsli
//
// The BVH node semantics, the watertight shear/permutation triangle test, the
// object-space traversal and the interpolated hit record match the HLSL
// software pipeline; only the byte-address buffers became typed views.

#include "sparkium/backends/core/buffer.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

struct RayDesc {
  float3 Origin;
  float3 Direction;
  float TMin;
  float TMax;
};

struct SoftwareHit {
  float distance;
  float2 barycentric;
  uint32_t instance;
  uint32_t primitive;
};

SPARKIUM_HD inline bool SoftwareBoxHit(const SoftwareNode &node,
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
      float a = (node.lo[axis] - origin[axis]) / direction[axis];
      float b = (node.hi[axis] - origin[axis]) / direction[axis];
      t_min = device::max(t_min, device::min(a, b));
      t_max = device::min(t_max, device::max(a, b));
      if (t_min > t_max)
        return false;
    }
  }
  return true;
}

SPARKIUM_HD inline bool SoftwareTriangleHit(const ByteBufferView &geometry,
                                            uint32_t primitive,
                                            const float3 &origin,
                                            const float3 &direction,
                                            float t_min,
                                            SoftwareHit &hit) {
  uint3 ids = geometry.Load3(geometry.Load(48) + primitive * 12);
  uint32_t offset = geometry.Load(8), stride = geometry.Load(12);
  float3 a = geometry.LoadFloat3(offset + stride * ids.x);
  float3 b = geometry.LoadFloat3(offset + stride * ids.y);
  float3 c = geometry.LoadFloat3(offset + stride * ids.z);
  // Watertight shear/permutation test; shared edges use the same arithmetic.
  float3 abs_dir = device::abs(direction);
  uint32_t kz = abs_dir.x > abs_dir.y ? 0 : 1;
  if (abs_dir.z > abs_dir[kz])
    kz = 2;
  if (direction[kz] == 0.0f)
    return false;
  uint32_t kx = (kz + 1) % 3, ky = (kx + 1) % 3;
  if (direction[kz] < 0.0f) {
    uint32_t swap_axis = kx;
    kx = ky;
    ky = swap_axis;
  }
  float sx = direction[kx] / direction[kz], sy = direction[ky] / direction[kz];
  float sz = 1.0f / direction[kz];
  a = a - origin;
  b = b - origin;
  c = c - origin;
  float ax = a[kx] - sx * a[kz], ay = a[ky] - sy * a[kz];
  float bx = b[kx] - sx * b[kz], by = b[ky] - sy * b[kz];
  float cx = c[kx] - sx * c[kz], cy = c[ky] - sy * c[kz];
  float u = cx * by - cy * bx;
  float v = ax * cy - ay * cx;
  float w = bx * ay - by * ax;
  if ((device::min(u, device::min(v, w)) < 0.0f) && (device::max(u, device::max(v, w)) > 0.0f))
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

struct MeshInstance {
  Mat4x3 object_to_world;
  Mat4x3 world_to_object;
};

SPARKIUM_HD inline bool SoftwareTraceMesh(const MeshInstance &instance,
                                          uint32_t instance_index,
                                          uint32_t mesh_root,
                                          const ByteBufferView &geometry,
                                          const RayDesc &ray,
                                          bool any_hit,
                                          SoftwareHit &hit,
                                          const SoftwareNode *nodes) {
  float3 origin = mul(instance.world_to_object, float4(ray.Origin, 1));
  // Do not normalize: the parameter t must remain in world-ray units under scaling.
  float3 direction = mul(instance.world_to_object, float4(ray.Direction, 0));
  uint32_t stack[32], size = 1;
  stack[0] = mesh_root;
  bool found = false;
  while (size != 0) {
    const SoftwareNode &node = nodes[stack[--size]];
    if (!SoftwareBoxHit(node, origin, direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SPARKIUM_SOFTWARE_INVALID) {
      if (node.second != SPARKIUM_SOFTWARE_INVALID &&
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

SPARKIUM_HD inline bool InlineIntersect(const DeviceScene &scene,
                                        const RayDesc &ray,
                                        bool any_hit,
                                        SoftwareHit &hit) {
  hit.distance = ray.TMax;
  hit.barycentric = float2(0.0f, 0.0f);
  hit.instance = SPARKIUM_SOFTWARE_INVALID;
  hit.primitive = SPARKIUM_SOFTWARE_INVALID;
  if (scene.num_instances == 0)
    return false;
  uint32_t stack[32], size = 1;
  stack[0] = 0;
  bool found = false;
  while (size != 0) {
    const SoftwareNode &node = scene.instance_nodes[stack[--size]];
    if (!SoftwareBoxHit(node, ray.Origin, ray.Direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SPARKIUM_SOFTWARE_INVALID) {
      if (node.second != SPARKIUM_SOFTWARE_INVALID) {
        const InstanceData &instance = scene.instances[node.second];
        MeshInstance transform{instance.object_to_world, instance.world_to_object};
        if (SoftwareTraceMesh(transform, node.second, scene.meshes[instance.mesh].root,
                              MeshBuffer(scene, instance.mesh), ray, any_hit, hit, scene.mesh_nodes)) {
          found = true;
          if (any_hit)
            return true;
        }
      }
    } else {
      stack[size++] = node.second;
      stack[size++] = node.first;
    }
  }
  return found;
}

// Port of MakeMeshHitRecord.
SPARKIUM_HD inline HitRecord MakeMeshHitRecord(uint32_t geometry_index,
                                              uint32_t object_index,
                                              uint32_t primitive_index,
                                              const float2 &barycentric,
                                              float distance,
                                              const float3 &ray_direction,
                                              const Mat4x3 &object_to_world,
                                              const Mat4x3 &world_to_object,
                                              const DeviceScene &scene) {
  ByteBufferView geometry = MeshBuffer(scene, geometry_index);
  uint32_t position_offset = geometry.Load(8), position_stride = geometry.Load(12);
  uint32_t normal_offset = geometry.Load(16), normal_stride = geometry.Load(20);
  uint32_t tex_coord_offset = geometry.Load(24), tex_coord_stride = geometry.Load(28);
  uint32_t tangent_offset = geometry.Load(32), tangent_stride = geometry.Load(36);
  uint32_t signal_offset = geometry.Load(40), signal_stride = geometry.Load(44);
  uint32_t index_offset = geometry.Load(48);
  uint32_t color_offset = geometry.Load(52), color_stride = geometry.Load(56);

  uint3 vid = geometry.Load3(index_offset + primitive_index * 12);

  float3 pos[3] = {geometry.LoadFloat3(position_offset + position_stride * vid.x),
                   geometry.LoadFloat3(position_offset + position_stride * vid.y),
                   geometry.LoadFloat3(position_offset + position_stride * vid.z)};

  float3 barycentrics = float3(1.0 - barycentric.x - barycentric.y, barycentric.x, barycentric.y);

  HitRecord hit_record;
  hit_record.t = distance;
  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.object_position = hit_record.position;
  hit_record.object_origin = mul(object_to_world, float4(0, 0, 0, 1));
  float3 world_pos[3] = {mul(object_to_world, float4(pos[0], 1.0f)), mul(object_to_world, float4(pos[1], 1.0f)),
                         mul(object_to_world, float4(pos[2], 1.0f))};
  hit_record.position =
      world_pos[0] * barycentrics[0] + world_pos[1] * barycentrics[1] + world_pos[2] * barycentrics[2];
  hit_record.geom_normal =
      device::normalize(mul(world_to_object, device::cross(pos[1] - pos[0], pos[2] - pos[0])));
  // Light-hit MIS is evaluated in world-space solid angle. Using the local
  // triangle area here loses most of the BSDF-sampled contribution for scaled
  // emitters (for example a unit quad scaled to a 30 x 30 area light).
  hit_record.pdf =
      1.0f / (device::length(device::cross(world_pos[1] - world_pos[0], world_pos[2] - world_pos[0])) * 0.5f);

  if (normal_offset != 0) {
    hit_record.normal = geometry.LoadFloat3(normal_offset + normal_stride * vid.x) * barycentrics[0] +
                        geometry.LoadFloat3(normal_offset + normal_stride * vid.y) * barycentrics[1] +
                        geometry.LoadFloat3(normal_offset + normal_stride * vid.z) * barycentrics[2];
    hit_record.normal = device::normalize(mul(world_to_object, hit_record.normal));
  } else {
    hit_record.normal = hit_record.geom_normal;
  }
  if (tex_coord_offset != 0) {
    hit_record.tex_coord = geometry.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.x) * barycentrics[0] +
                           geometry.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.y) * barycentrics[1] +
                           geometry.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.z) * barycentrics[2];
  } else {
    hit_record.tex_coord = float2(0.0, 0.0);
  }
  if (color_offset != 0) {
    hit_record.color = geometry.LoadFloat3(color_offset + color_stride * vid.x) * barycentrics[0] +
                       geometry.LoadFloat3(color_offset + color_stride * vid.y) * barycentrics[1] +
                       geometry.LoadFloat3(color_offset + color_stride * vid.z) * barycentrics[2];
  } else {
    hit_record.color = float3(1.0, 1.0, 1.0);
  }

  if (tangent_offset != 0) {
    hit_record.tangent = geometry.LoadFloat3(tangent_offset + tangent_stride * vid.x) * barycentrics[0] +
                         geometry.LoadFloat3(tangent_offset + tangent_stride * vid.y) * barycentrics[1] +
                         geometry.LoadFloat3(tangent_offset + tangent_stride * vid.z) * barycentrics[2];
    hit_record.tangent = device::normalize(mul(object_to_world, hit_record.tangent));
  } else {
    hit_record.tangent = device::cross(hit_record.normal, float3(0.0, 0.0, 1.0));
    if (device::length(hit_record.tangent) < 0.001) {
      hit_record.tangent = device::cross(hit_record.normal, float3(1.0, 0.0, 0.0));
    }
    hit_record.tangent = device::normalize(hit_record.tangent);
  }

  if (signal_offset != 0) {
    hit_record.signal = geometry.LoadFloat(signal_offset + signal_stride * vid.x) * barycentrics[0] +
                        geometry.LoadFloat(signal_offset + signal_stride * vid.y) * barycentrics[1] +
                        geometry.LoadFloat(signal_offset + signal_stride * vid.z) * barycentrics[2];
  } else {
    // Zero marks the absence of a valid tangent frame. Tangent-space normal
    // maps must then leave the interpolated surface normal unchanged.
    hit_record.signal = 0.0;
  }

  if (device::dot(ray_direction, hit_record.normal) > 0.0) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = static_cast<int32_t>(object_index);
  hit_record.primitive_index = static_cast<int32_t>(primitive_index);

  return hit_record;
}

// Port of MeshSamplePrimitive.
SPARKIUM_HD inline GeometryPrimitiveSample MeshSamplePrimitive(const ByteBufferView &geometry_data,
                                                              const Mat4x3 &transform,
                                                              uint32_t primitive_id,
                                                              float2 sample) {
  uint32_t position_offset = geometry_data.Load(8);
  uint32_t position_stride = geometry_data.Load(12);
  uint32_t index_offset = geometry_data.Load(48);
  uint32_t tex_coord_offset = geometry_data.Load(24);
  uint32_t tex_coord_stride = geometry_data.Load(28);
  uint3 vid = geometry_data.Load3(index_offset + primitive_id * 3 * 4);
  if (sample.x + sample.y > 1.0f) {
    // Handle case where sample is outside the triangle
    sample = float2(1.0f - sample.x, 1.0f - sample.y);
  }
  float3 barycentrics = float3(1.0f - sample.x - sample.y, sample.x, sample.y);
  float3 pos[3];
  pos[0] = mul(transform, float4(geometry_data.LoadFloat3(position_offset + position_stride * vid.x), 1.0f));
  pos[1] = mul(transform, float4(geometry_data.LoadFloat3(position_offset + position_stride * vid.y), 1.0f));
  pos[2] = mul(transform, float4(geometry_data.LoadFloat3(position_offset + position_stride * vid.z), 1.0f));

  GeometryPrimitiveSample sample_result;
  sample_result.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  sample_result.normal = device::normalize(device::cross(pos[1] - pos[0], pos[2] - pos[0]));
  if (tex_coord_offset != 0) {
    sample_result.tex_coord =
        geometry_data.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.x) * barycentrics[0] +
        geometry_data.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.y) * barycentrics[1] +
        geometry_data.LoadFloat2(tex_coord_offset + tex_coord_stride * vid.z) * barycentrics[2];
  } else {
    sample_result.tex_coord = float2(0.0f, 0.0f);
  }
  sample_result.pdf = 1.0f / (device::length(device::cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);
  return sample_result;
}

}  // namespace sparkium::backends
