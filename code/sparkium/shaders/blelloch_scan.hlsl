#include "compute_contract.hlsli"

SP_RESOURCE(RWByteAddressBuffer, buffer, u0, 0);
#define SP_BINDING_buffer SP_RESOURCE_ACCESS(RWByteAddressBuffer, buffer, 0)

struct Metadata {
  uint offset;
  uint stride;
  uint element_count;
};

SP_RESOURCE(ConstantBuffer<Metadata>, metadata, b0, 1);
#define SP_BINDING_metadata SP_RESOURCE_ACCESS(ConstantBuffer<Metadata>, metadata, 1)

#define GROUP_SIZE 64
#ifndef SPARKIUM_CPU
groupshared float group_element[GROUP_SIZE];
#endif

SP_NUMTHREADS(GROUP_SIZE, 1, 1)

void BlellochUpSweep(SP_CONTEXT uint3 DTID : SV_DispatchThreadID, uint3 GTID : SV_GroupThreadID) {
#ifdef SPARKIUM_CPU
  if (GTID.x != 0)
    return;
  float prefix = 0.0f;
  uint first = DTID.x;
  for (uint i = first; i < min(first + GROUP_SIZE, SP_BINDING_metadata.element_count); ++i) {
    uint address = SP_BINDING_metadata.offset + i * SP_BINDING_metadata.stride;
    prefix += asfloat(SP_BINDING_buffer.Load(address));
    SP_BINDING_buffer.Store(address, asuint(prefix));
  }
#else
  uint index = DTID.x;
  float element = 0.0;
  if (index < SP_BINDING_metadata.element_count) {
    element = asfloat(SP_BINDING_buffer.Load(SP_BINDING_metadata.offset + index * SP_BINDING_metadata.stride));
  }
  element += WavePrefixSum(element);

  group_element[GTID.x] = element;
  GroupMemoryBarrierWithGroupSync();
  for (uint prefix_range = WaveGetLaneCount() * 2; prefix_range <= GROUP_SIZE; prefix_range *= 2) {
    if (GTID.x % prefix_range >= prefix_range / 2) {
      group_element[GTID.x] += group_element[GTID.x / prefix_range * prefix_range + prefix_range / 2 - 1];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  element = group_element[GTID.x];

  if (index < SP_BINDING_metadata.element_count) {
    SP_BINDING_buffer.Store(SP_BINDING_metadata.offset + index * SP_BINDING_metadata.stride, asuint(element));
  }
#endif
}

SP_NUMTHREADS(GROUP_SIZE, 1, 1) void BlellochDownSweep(SP_CONTEXT uint3 DTID : SV_DispatchThreadID) {
  uint index = DTID.x;
  if (index / GROUP_SIZE) {
    float element = 0.0f;
    if (index < SP_BINDING_metadata.element_count) {
      element = asfloat(SP_BINDING_buffer.Load(SP_BINDING_metadata.offset + index * SP_BINDING_metadata.stride));
    }
    uint add_index = (index / GROUP_SIZE - 1) * GROUP_SIZE + GROUP_SIZE - 1;
    float added_element = 0.0f;
    if (add_index < SP_BINDING_metadata.element_count) {
      added_element =
          asfloat(SP_BINDING_buffer.Load(SP_BINDING_metadata.offset + add_index * SP_BINDING_metadata.stride));
    }
    element += added_element;
    if (index < SP_BINDING_metadata.element_count && index % GROUP_SIZE != GROUP_SIZE - 1) {
      SP_BINDING_buffer.Store(SP_BINDING_metadata.offset + index * SP_BINDING_metadata.stride, asuint(element));
    }
  }
}
