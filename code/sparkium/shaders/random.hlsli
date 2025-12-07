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

uint SobolUint(inout RandomDevice random_device) {
 uint result = sobol_table.Load((random_device.samp * 1024 + (random_device.dim)) * 4)
 ^
WangHash(random_device.offset);
random_device.dim++;
return result;
}

uint RandomUint(inout RandomDevice random_device) {
  if (random_device.dim < 1024 && random_device.samp < 65536)
    return SobolUint(random_device);
  return WangHash(random_device.seed);
}

float RandomFloat(inout RandomDevice rd) {
  return float(RandomUint(rd)) / 4294967296.0;
}

float2 RandomOnCircle(inout RandomDevice rd) {
  float theta = RandomFloat(rd) * PI * 2.0;
  return float2(sin(theta), cos(theta));
}

float2 RandomInCircle(inout RandomDevice rd) {
  return RandomOnCircle(rd) * sqrt(RandomFloat(rd));
}

float3 RandomOnSphere(inout RandomDevice rd) {
  float z = RandomFloat(rd) * 2.0 - 1.0;
  float xy = sqrt(1.0 - z * z);
  return float3(xy * RandomOnCircle(rd), z);
}

float3 RandomInSphere(inout RandomDevice rd) {
  return RandomOnSphere(rd) * pow(RandomFloat(rd), 0.3333333333333333333);
}

void SampleCosHemisphere(inout RandomDevice rd, const float3 N, out float3 omega_in, out float pdf) {
  sample_cos_hemisphere(N, RandomFloat(rd), RandomFloat(rd), omega_in, pdf);
}
