RWStructuredBuffer<float> values;

[numthreads(1, 1, 1)] void computeMain(uint3 id : SV_DispatchThreadID) { values[id.x] = values[id.x] * 2.0f + 1.0f; }
