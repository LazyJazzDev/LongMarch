#include "native_contract.hlsli"
#include "buffer_helper.hlsli"
#include "common.hlsli"

SP_RESOURCE(ByteAddressBuffer, light_metadatas, t0, 0);
#define SP_BINDING_light_metadatas SP_RESOURCE_ACCESS(ByteAddressBuffer, light_metadatas, 0)
SP_RESOURCE(RWByteAddressBuffer, light_selector_data, u0, 1);
#define SP_BINDING_light_selector_data SP_RESOURCE_ACCESS(RWByteAddressBuffer, light_selector_data, 1)
SP_ARRAY_RESOURCE(ByteAddressBuffer, data_buffers, t0, 2);
#define SP_BINDING_data_buffers SP_ARRAY_ACCESS(ByteAddressBuffer, data_buffers, 2)

#define GROUP_SIZE 64
#ifndef SPARKIUM_NATIVE_CPU
groupshared float group_element[GROUP_SIZE];
#endif

SP_NUMTHREADS(GROUP_SIZE, 1, 1)

void GatherLightPowerKernel(SP_CONTEXT uint3 DTID : SV_DispatchThreadID, uint3 GTID : SV_GroupThreadID) {
  uint light_count = SP_BINDING_light_selector_data.Load(0);
  BufferReference SP_BUFFER_ARG(RWByteAddressBuffer) power_pdf = MakeBufferReference(SP_BINDING_light_selector_data, 4);
#ifdef SPARKIUM_NATIVE_CPU
  if (GTID.x != 0)
    return;
  float prefix = 0.0f;
  for (uint i = DTID.x; i < min(DTID.x + GROUP_SIZE, light_count); ++i) {
    LightMetadata SP_BINDING_metadata = SP_BINDING_light_metadatas.Load<LightMetadata>(sizeof(LightMetadata) * i);
    prefix +=
        SP_BINDING_data_buffers[SP_BINDING_metadata.sampler_data_index].Load<float>(SP_BINDING_metadata.power_offset);
    power_pdf.Store(i * 4, asuint(prefix));
  }
#else
  float power = 0.0f;
  if (DTID.x < light_count) {
    LightMetadata SP_BINDING_metadata = SP_BINDING_light_metadatas.Load<LightMetadata>(sizeof(LightMetadata) * DTID.x);
    uint light_sampler_data_index = SP_BINDING_metadata.sampler_data_index;
    uint power_offset = SP_BINDING_metadata.power_offset;
    power = SP_BINDING_data_buffers[SP_NONUNIFORM(light_sampler_data_index)].Load<float>(power_offset);
  }
  power += WavePrefixSum(power);

  group_element[GTID.x] = power;
  GroupMemoryBarrierWithGroupSync();
  for (uint prefix_range = WaveGetLaneCount() * 2; prefix_range <= GROUP_SIZE; prefix_range *= 2) {
    if (GTID.x % prefix_range >= prefix_range / 2) {
      group_element[GTID.x] += group_element[GTID.x / prefix_range * prefix_range + prefix_range / 2 - 1];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  power = group_element[GTID.x];

  // The last workgroup can contain inactive lanes. Metal argument-buffer
  // pointers do not provide Vulkan's robust out-of-bounds buffer accesses.
  if (DTID.x < light_count) {
    power_pdf.Store(DTID.x * 4, asuint(power));
  }
#endif
}
