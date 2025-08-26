#include "buffer_helper.hlsli"
#include "common.hlsli"
#include "constants.hlsli"

#define GROUP_SIZE 128

ByteAddressBuffer geometry_data : register(t0, space0);
ByteAddressBuffer material_data : register(t0, space1);
RWByteAddressBuffer direct_lighting_sampler_data : register(u0, space2);

// clang-format off
#include "geometry_sampler.hlsli"
#include "material_evaluator.hlsli"
#include "material_power_sampler.hlsli"
// clang-format on

[numthreads(GROUP_SIZE, 1, 1)] void GatherPrimitivePowerKernel(uint3 GID
                                                               : SV_GroupID, uint3 DTID
                                                               : SV_DispatchThreadID, uint3 GTID
                                                               : SV_GroupThreadID) {
  float3x4 transform = LoadFloat3x4(direct_lighting_sampler_data, 0);
  uint primitive_count = direct_lighting_sampler_data.Load(48);
  BufferReference<RWByteAddressBuffer> power_pdf = MakeBufferReference(direct_lighting_sampler_data, 52);
  float primitive_power = 0.0f;
  // calculate the prefix sum of primitive_power_shared with WavePrefixSum
  if (DTID.x < primitive_count) {
    primitive_power = PowerSampler(material_data, geometry_data, transform, DTID.x);
  }

  primitive_power += WavePrefixSum(primitive_power);

  if (DTID.x < primitive_count) {
    power_pdf.Store(DTID.x * 4, asuint(primitive_power));
  }
}
