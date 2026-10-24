#include "common.slang"
#include "types.slang"
RHI_UNIFORM(float, gain)

struct Input {
  RHI_INPUT(float3, position, POSITION)
};

float4 VSMain(Input input) : SV_Position {
  return float4(input.position * gain * GetValue(), 1.0);
}
