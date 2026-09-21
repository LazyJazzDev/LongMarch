#include "compute_contract.hlsli"
#pragma once
#include "constants.hlsli"
#include "bindings.hlsli"

uint WangHash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

uint WangHashS(uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

RandomDevice InitRandomSeed(uint x, uint y, uint s) {
  RandomDevice random_device;
  random_device.offset = WangHashS(WangHashS(x) ^ y);
  random_device.seed = WangHashS(random_device.offset ^ s);
  random_device.dim = 0;
  random_device.samp = s;
  return random_device;
}

uint SobolUint(SP_CONTEXT inout RandomDevice random_device) {
  uint result = SP_BINDING_sobol_table.Load((random_device.samp * 1024 + (random_device.dim)) * 4) ^
                WangHash(random_device.offset);
  random_device.dim++;
  return result;
}

uint RandomUint(SP_CONTEXT inout RandomDevice random_device) {
  if (random_device.dim < 1024 && random_device.samp < 65536)
    return SobolUint(SP_CONTEXT_ARG random_device);
  return WangHash(random_device.seed);
}

float RandomFloat(SP_CONTEXT inout RandomDevice rd) {
  return float(RandomUint(SP_CONTEXT_ARG rd)) / 4294967296.0;
}

float2 RandomOnCircle(SP_CONTEXT inout RandomDevice rd) {
  float theta = RandomFloat(SP_CONTEXT_ARG rd) * PI * 2.0;
  return float2(sin(theta), cos(theta));
}

float2 RandomInCircle(SP_CONTEXT inout RandomDevice rd) {
  return RandomOnCircle(SP_CONTEXT_ARG rd) * sqrt(RandomFloat(SP_CONTEXT_ARG rd));
}

float3 RandomOnSphere(SP_CONTEXT inout RandomDevice rd) {
  float z = RandomFloat(SP_CONTEXT_ARG rd) * 2.0 - 1.0;
  float xy = sqrt(1.0 - z * z);
  return float3(xy * RandomOnCircle(SP_CONTEXT_ARG rd), z);
}

float3 RandomInSphere(SP_CONTEXT inout RandomDevice rd) {
  return RandomOnSphere(SP_CONTEXT_ARG rd) * pow(RandomFloat(SP_CONTEXT_ARG rd), 0.3333333333333333333);
}

void SampleCosHemisphere(SP_CONTEXT inout RandomDevice rd, const float3 N, out float3 omega_in, out float pdf) {
  sample_cos_hemisphere(N, RandomFloat(SP_CONTEXT_ARG rd), RandomFloat(SP_CONTEXT_ARG rd), omega_in, pdf);
}
