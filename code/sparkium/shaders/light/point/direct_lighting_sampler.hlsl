#include "compute_contract.hlsli"
#include "light/point/sampler.hlsli"

// callable shader to sample direct lighting
[shader("callable")] void SampleDirectLightingCallable(SP_CONTEXT inout SampleDirectLightingPayload payload) {
  PointLightSampler(SP_CONTEXT_ARG payload);
}
