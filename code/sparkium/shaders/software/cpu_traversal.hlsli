#pragma once
#include "software/layout.hlsli"
#include "software/intersection.hlsli"

// CPU SAH tree: adjacent children, up to four primitives per normal leaf.
// This layout and traversal policy are independent of the GPU heap BVH.
struct CpuNode {
  float3 lo;
  uint first;
  float3 hi;
  uint count;
};

CpuNode CpuLoadNode(SP_CONTEXT uint tree, uint index) {
  return SP_BINDING_software_nodes.Load<CpuNode>(tree + 16 + index * 32);
}

bool CpuBox(CpuNode node,
            float3 origin,
            float3 direction,
            float3 inverse_direction,
            float t_min,
            float t_max,
            out float near_t) {
  near_t = t_min;
  if (any(node.lo > node.hi))
    return false;
  for (uint axis = 0; axis < 3; ++axis) {
    if (direction[axis] == 0) {
      if (origin[axis] < node.lo[axis] || origin[axis] > node.hi[axis])
        return false;
    } else {
      float a = (node.lo[axis] - origin[axis]) * inverse_direction[axis];
      float b = (node.hi[axis] - origin[axis]) * inverse_direction[axis];
      near_t = max(near_t, min(a, b));
      t_max = min(t_max, max(a, b));
      if (near_t > t_max)
        return false;
    }
  }
  return true;
}

uint CpuPrimitive(SP_CONTEXT uint tree, uint index) {
  return SP_BINDING_software_nodes.Load(tree + SP_BINDING_software_nodes.Load(tree + 4) + index * 4);
}

struct CpuChildren {
  uint2 nodes;
  float2 distances;
  uint count;
};

CpuChildren CpuChildHits(SP_CONTEXT CpuNode node,
                         uint tree,
                         float3 origin,
                         float3 direction,
                         float3 inverse_direction,
                         float t_min,
                         float t_max) {
  CpuChildren result = (CpuChildren)0;
  float a, b;
  bool left =
      CpuBox(CpuLoadNode(SP_CONTEXT_ARG tree, node.first), origin, direction, inverse_direction, t_min, t_max, a);
  bool right =
      CpuBox(CpuLoadNode(SP_CONTEXT_ARG tree, node.first + 1), origin, direction, inverse_direction, t_min, t_max, b);
  if (left && right) {
    result.nodes = a <= b ? uint2(node.first + 1, node.first) : uint2(node.first, node.first + 1);
    result.distances = a <= b ? float2(b, a) : float2(a, b);
    result.count = 2;
  } else if (left || right) {
    result.nodes.x = left ? node.first : node.first + 1;
    result.distances.x = left ? a : b;
    result.count = 1;
  }
  return result;
}

bool CpuTraceMesh(SP_CONTEXT uint instance_index, SP_RAY ray, bool any_hit, inout SoftwareHit hit) {
  SoftwareInstance instance = LoadSoftwareInstance(SP_BINDING_software_instances, instance_index);
  float3 origin = mul(instance.world_to_object, float4(ray.Origin, 1));
  float3 direction = mul(instance.world_to_object, float4(ray.Direction, 0));
  float3 inverse_direction = 1.0f / direction;
  ByteAddressBuffer geometry = SP_BINDING_data_buffers[instance.geometry];
  uint stack[64], size = 1;
  float distances[64];
  stack[0] = 0;
  if (!CpuBox(CpuLoadNode(SP_CONTEXT_ARG instance.root, 0), origin, direction, inverse_direction, ray.TMin,
              hit.distance, distances[0]))
    return false;
  bool found = false;
  while (size != 0) {
    --size;
    if (distances[size] > hit.distance)
      continue;
    CpuNode node = CpuLoadNode(SP_CONTEXT_ARG instance.root, stack[size]);
    if (node.count != 0) {
      for (uint i = 0; i < node.count; ++i) {
        uint primitive = CpuPrimitive(SP_CONTEXT_ARG instance.root, node.first + i);
        if (SoftwareTriangleHit(geometry, primitive, origin, direction, ray.TMin, hit)) {
          found = true;
          hit.instance = instance_index;
          if (any_hit)
            return true;
        }
      }
    } else {
      CpuChildren children = CpuChildHits(SP_CONTEXT_ARG node, instance.root, origin, direction, inverse_direction,
                                          ray.TMin, hit.distance);
      for (uint i = 0; i < children.count; ++i) {
        stack[size] = children.nodes[i];
        distances[size++] = children.distances[i];
      }
    }
  }
  return found;
}

bool InlineIntersect(SP_CONTEXT SP_RAY ray, bool any_hit, out SoftwareHit hit) {
  hit = (SoftwareHit)0;
  hit.distance = ray.TMax;
  hit.instance = hit.primitive = SOFTWARE_INVALID;
  if (SP_BINDING_software_instances.Load(0) == 0)
    return false;
  float3 inverse_direction = 1.0f / ray.Direction;
  uint stack[64], size = 1;
  float distances[64];
  stack[0] = 0;
  if (!CpuBox(CpuLoadNode(SP_CONTEXT_ARG 0, 0), ray.Origin, ray.Direction, inverse_direction, ray.TMin, hit.distance,
              distances[0]))
    return false;
  bool found = false;
  while (size != 0) {
    --size;
    if (distances[size] > hit.distance)
      continue;
    CpuNode node = CpuLoadNode(SP_CONTEXT_ARG 0, stack[size]);
    if (node.count != 0) {
      for (uint i = 0; i < node.count; ++i)
        if (CpuTraceMesh(SP_CONTEXT_ARG CpuPrimitive(SP_CONTEXT_ARG 0, node.first + i), ray, any_hit, hit)) {
          found = true;
          if (any_hit)
            return true;
        }
    } else {
      CpuChildren children =
          CpuChildHits(SP_CONTEXT_ARG node, 0, ray.Origin, ray.Direction, inverse_direction, ray.TMin, hit.distance);
      for (uint i = 0; i < children.count; ++i) {
        stack[size] = children.nodes[i];
        distances[size++] = children.distances[i];
      }
    }
  }
  return found;
}
