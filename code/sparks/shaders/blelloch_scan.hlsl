
RWByteAddressBuffer buffer : register(u0, space0);

struct Metadata {
  uint offset;
  uint stride;
  uint element_count;
};

ConstantBuffer<Metadata> metadata : register(b0, space1);

[numthreads(64, 1, 1)] void BlellochUpSweep(uint3 DTID
                                            : SV_DispatchThreadID) {
  uint index = DTID.x;
  float element = 0.0;
  if (index < metadata.element_count) {
    element = asfloat(buffer.Load(metadata.offset + index * metadata.stride));
  }
  element += WavePrefixSum(element);
  if (index < metadata.element_count) {
    buffer.Store(metadata.offset + index * metadata.stride, asuint(element));
  }
}

    [numthreads(64, 1, 1)] void BlellochDownSweep(uint3 DTID
                                                  : SV_DispatchThreadID) {
  uint index = DTID.x;
  if (index / WaveGetLaneCount()) {
    float element = 0.0f;
    if (index < metadata.element_count) {
      element = asfloat(buffer.Load(metadata.offset + index * metadata.stride));
    }
    uint add_index = (index / WaveGetLaneCount() - 1) * WaveGetLaneCount() + WaveGetLaneCount() - 1;
    element += asfloat(buffer.Load(metadata.offset + add_index * metadata.stride));
    if (index < metadata.element_count && index % WaveGetLaneCount() != WaveGetLaneCount() - 1) {
      buffer.Store(metadata.offset + index * metadata.stride, asuint(element));
    }
  }  //*/
}
