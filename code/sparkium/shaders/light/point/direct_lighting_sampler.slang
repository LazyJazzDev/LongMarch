#include "light/point/sampler.slang"

// callable shader to sample direct lighting
[shader("callable")] void SampleDirectLightingCallable(inout SampleDirectLightingPayload payload) {
  PointLightSampler(payload);
}
