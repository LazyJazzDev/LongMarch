#include "native_contract.hlsli"
#pragma once
#include "software/layout.hlsli"
#ifndef SOFTWARE_EXTERNAL_BINDINGS
#include "bindings.hlsli"
#else
#define SP_BINDING_data_buffers data_buffers
#define SP_BINDING_software_nodes software_nodes
#define SP_BINDING_software_instances software_instances
#endif

#include "software/intersection.hlsli"

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

bool SoftwareTraceMesh(SP_CONTEXT SoftwareInstance instance,
                       uint instance_index,
                       SP_RAY ray,
                       bool any_hit,
                       inout SoftwareHit hit) {
  float3 origin = mul(instance.world_to_object, float4(ray.Origin, 1));
  // Do not normalize: the parameter t must remain in world-ray units under scaling.
  float3 direction = mul(instance.world_to_object, float4(ray.Direction, 0));
  ByteAddressBuffer geometry = SP_BINDING_data_buffers[SP_NONUNIFORM(instance.geometry)];
  uint stack[32], size = 1;
  stack[0] = instance.root;
  bool found = false;
  while (size != 0) {
    SoftwareNode node = LoadSoftwareNode(SP_BINDING_software_nodes, stack[--size]);
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

bool InlineIntersect(SP_CONTEXT SP_RAY ray, bool any_hit, out SoftwareHit hit) {
  hit = (SoftwareHit)0;
  hit.distance = ray.TMax;
  hit.instance = hit.primitive = SOFTWARE_INVALID;
  if (SP_BINDING_software_instances.Load(0) == 0)
    return false;
  uint stack[32], size = 1;
  stack[0] = 0;
  bool found = false;
  while (size != 0) {
    SoftwareNode node = LoadSoftwareNode(SP_BINDING_software_nodes, stack[--size]);
    if (!SoftwareBoxHit(node, ray.Origin, ray.Direction, ray.TMin, hit.distance))
      continue;
    if (node.first == SOFTWARE_INVALID) {
      if (node.second != SOFTWARE_INVALID &&
          SoftwareTraceMesh(SP_CONTEXT_ARG LoadSoftwareInstance(SP_BINDING_software_instances, node.second),
                            node.second, ray, any_hit, hit)) {
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
