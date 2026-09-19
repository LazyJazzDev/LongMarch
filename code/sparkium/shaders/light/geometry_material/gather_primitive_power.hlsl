#include "native_contract.hlsli"
#include "buffer_helper.hlsli"
#include "common.hlsli"
#include "constants.hlsli"

#define GROUP_SIZE 64
#ifndef SPARKIUM_NATIVE_CPU
groupshared float group_element[GROUP_SIZE];
#endif

SP_RESOURCE(ByteAddressBuffer, geometry_data, t0, 0);
#define SP_BINDING_geometry_data SP_RESOURCE_ACCESS(ByteAddressBuffer, geometry_data, 0)
SP_RESOURCE(ByteAddressBuffer, material_data, t0, 1);
#define SP_BINDING_material_data SP_RESOURCE_ACCESS(ByteAddressBuffer, material_data, 1)
SP_RESOURCE(RWByteAddressBuffer, direct_lighting_sampler_data, u0, 2);
#define SP_BINDING_direct_lighting_sampler_data SP_RESOURCE_ACCESS(RWByteAddressBuffer, direct_lighting_sampler_data, 2)

// clang-format off
#include "geometry_sampler.hlsli"
#include "material_evaluator.hlsli"
// clang-format on

SP_NUMTHREADS(GROUP_SIZE, 1, 1)

void GatherPrimitivePowerKernel(SP_CONTEXT uint3 GID
                                : SV_GroupID, uint3 DTID
                                : SV_DispatchThreadID, uint3 GTID
                                : SV_GroupThreadID) {
  float3x4 transform = LoadFloat3x4(SP_BINDING_direct_lighting_sampler_data, 0);
  uint primitive_count = SP_BINDING_direct_lighting_sampler_data.Load(48);
  BufferReference SP_BUFFER_ARG(RWByteAddressBuffer) power_pdf =
      MakeBufferReference(SP_BINDING_direct_lighting_sampler_data, 52);
  float primitive_power = 0.0f;
  GeometrySampler SP_BUFFER_ARG(ByteAddressBuffer) geometry_sampler;
  geometry_sampler.geometry_data = SP_BINDING_geometry_data;
  geometry_sampler.SetTransform(transform);
  MaterialEvaluator SP_BUFFER_ARG(ByteAddressBuffer) material_evaluator;
  material_evaluator.material_data = SP_BINDING_material_data;
#ifdef SPARKIUM_NATIVE_CPU
  if (GTID.x != 0)
    return;
  float prefix = 0.0f;
  for (uint i = DTID.x; i < min(DTID.x + GROUP_SIZE, primitive_count); ++i) {
    prefix += material_evaluator.PrimitivePower(geometry_sampler, i);
    power_pdf.Store(i * 4, asuint(prefix));
  }
#else
  // calculate the prefix sum of primitive_power_shared with WavePrefixSum
  if (DTID.x < primitive_count) {
    primitive_power = material_evaluator.PrimitivePower(geometry_sampler, DTID.x);
  }

  primitive_power += WavePrefixSum(primitive_power);

  group_element[GTID.x] = primitive_power;
  GroupMemoryBarrierWithGroupSync();
  for (uint prefix_range = WaveGetLaneCount() * 2; prefix_range <= GROUP_SIZE; prefix_range *= 2) {
    if (GTID.x % prefix_range >= prefix_range / 2) {
      group_element[GTID.x] += group_element[GTID.x / prefix_range * prefix_range + prefix_range / 2 - 1];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  primitive_power = group_element[GTID.x];

  if (DTID.x < primitive_count) {
    power_pdf.Store(DTID.x * 4, asuint(primitive_power));
  }
#endif
}
