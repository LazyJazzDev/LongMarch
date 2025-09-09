#include "light/point/sampler.hlsli"

// callable shader to sample direct lighting
[shader("callable")] void SampleDirectLightingCallable(inout SampleDirectLightingPayload payload) {
  PointLightSampler(payload);
}
