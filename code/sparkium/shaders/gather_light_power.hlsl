#include "buffer_helper.hlsli"
#include "common.hlsli"

ByteAddressBuffer light_metadatas : register(t0, space0);
RWByteAddressBuffer light_selector_data : register(u0, space1);
ByteAddressBuffer data_buffers[] : register(t0, space2);

[numthreads(64, 1, 1)] void GatherLightPowerKernel(uint3 DTID
                                                   : SV_DispatchThreadID) {
  uint light_count = light_selector_data.Load(0);
  BufferReference<RWByteAddressBuffer> power_pdf = MakeBufferReference(light_selector_data, 4);
  uint light_sampler_data_index =
      light_metadatas.Load<LightMetadata>(sizeof(LightMetadata) * DTID.x).sampler_data_index;
  uint power_offset = light_metadatas.Load<LightMetadata>(sizeof(LightMetadata) * DTID.x).power_offset;
  float power = 0.0f;
  if (DTID.x < light_count) {
    power = data_buffers[NonUniformResourceIndex(light_sampler_data_index)].Load<float>(power_offset);
  }
  power += WavePrefixSum(power);
  power_pdf.Store(DTID.x * 4, asuint(power));
}
