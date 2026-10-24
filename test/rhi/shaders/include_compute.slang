#include "include_settings.slang"
[numthreads(GROUP_SIZE, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {}
