#pragma once
#include "buffer_helper.hlsli"

// Explicit byte layouts shared with SoftwarePipeline. All indices are node indices.
static const uint SOFTWARE_NODE_BYTES = 32;
static const uint SOFTWARE_INSTANCE_BYTES = 112;
static const uint SOFTWARE_INVALID = 0xffffffff;

struct SoftwareNode {
  float3 lo;
  uint first;
  float3 hi;
  uint second;
};

template <class B>
SoftwareNode LoadSoftwareNode(B nodes, uint index) {
  return nodes.template Load<SoftwareNode>(index * SOFTWARE_NODE_BYTES);
}

struct SoftwareInstance {
  float3x4 object_to_world;
  float3x4 world_to_object;
  uint root;
  uint geometry;
  uint material;
  uint primitive_count;
};

SoftwareInstance LoadSoftwareInstance(ByteAddressBuffer instances, uint index) {
  uint offset = 16 + index * SOFTWARE_INSTANCE_BYTES;
  SoftwareInstance result;
  result.object_to_world = LoadFloat3x4(instances, offset);
  result.world_to_object = LoadFloat3x4(instances, offset + 48);
  uint4 info = instances.Load4(offset + 96);
  result.root = info.x;
  result.geometry = info.y;
  result.material = info.z;
  result.primitive_count = info.w;
  return result;
}
