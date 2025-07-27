#define PI 3.14159265358979323
#define INV_PI 0.31830988618379067
#define T_MIN 0.0001
#define T_MAX 10000.0

struct RandomDevice {
  uint offset;
  uint samp;
  uint seed;
  uint dim;
} random_device;

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


uint RandomUint(inout RandomDevice rd) {
  //  if (random_device.dim < 1024 && random_device.samp < 65536)
  //    return SobolUint();
  return WangHash(rd.seed);
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

void MakeOrthonormals(const float3 N, out float3 a, out float3 b) {
  if (N.x != N.y || N.x != N.z)
    a = float3(N.z - N.y, N.x - N.z, N.y - N.x);
  else
    a = float3(N.z - N.y, N.x + N.z, -N.y - N.x);

  a = normalize(a);
  b = cross(N, a);
}

void sample_cos_hemisphere(const float3 N,
                           float r1,
                           const float r2,
                           out float3 omega_in,
                           out float pdf) {
  r1 *= PI * 2.0;
  float3 T, B;
  MakeOrthonormals(N, T, B);
  omega_in = float3(float2(sin(r1), cos(r1)) * sqrt(1.0 - r2), sqrt(r2));
  pdf = omega_in.z * INV_PI;
  omega_in = mul(omega_in, float3x3(T, B, N));
}

void SampleCosHemisphere(inout RandomDevice rd, const float3 N, out float3 omega_in, out float pdf) {
  sample_cos_hemisphere(N, RandomFloat(rd), RandomFloat(rd), omega_in, pdf);
}
