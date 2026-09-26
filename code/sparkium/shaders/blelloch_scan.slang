
RWByteAddressBuffer buffer : register(u0, space0);

struct Metadata {
  uint offset;
  uint stride;
  uint element_count;
};

ConstantBuffer<Metadata> metadata : register(b0, space1);

#define GROUP_SIZE 64
groupshared float group_element[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)] void BlellochUpSweep(uint3 DTID
                                                    : SV_DispatchThreadID, uint3 GTID
                                                    : SV_GroupThreadID) {
  uint index = DTID.x;
  float element = 0.0;
  if (index < metadata.element_count) {
    element = asfloat(buffer.Load(metadata.offset + index * metadata.stride));
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

  if (index < metadata.element_count) {
    buffer.Store(metadata.offset + index * metadata.stride, asuint(element));
  }
}

    [numthreads(GROUP_SIZE, 1, 1)] void BlellochDownSweep(uint3 DTID
                                                          : SV_DispatchThreadID) {
  uint index = DTID.x;
  if (index / GROUP_SIZE) {
    float element = 0.0f;
    if (index < metadata.element_count) {
      element = asfloat(buffer.Load(metadata.offset + index * metadata.stride));
    }
    uint add_index = (index / GROUP_SIZE - 1) * GROUP_SIZE + GROUP_SIZE - 1;
    float added_element = 0.0f;
    if (add_index < metadata.element_count) {
      added_element = asfloat(buffer.Load(metadata.offset + add_index * metadata.stride));
    }
    element += added_element;
    if (index < metadata.element_count && index % GROUP_SIZE != GROUP_SIZE - 1) {
      buffer.Store(metadata.offset + index * metadata.stride, asuint(element));
    }
  }
}
