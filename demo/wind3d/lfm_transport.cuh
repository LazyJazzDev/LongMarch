#pragma once

// Included inside simulation.cu's private namespace, after its MAC samplers.
// Sun et al., Leapfrog Flow Maps (SIGGRAPH 2025), sections 4.1-4.3.
// Coordinates and map derivatives here are in grid units. Unlike the reference
// tiled implementation, fields use a Cartesian MAC layout with wall extensions.
constexpr int lfm_intervals = 5;

__global__ void LfmAccelerationDivergence(Grid g, float *predicted, const float *current) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n < g.nx * g.ny * g.nz)
    predicted[n] -= current[n];
}

struct LfmVelocity {
  const float *axis[3];
};

struct LfmHistory {
  LfmVelocity velocity[lfm_intervals];
};

__device__ Vec3 LfmOffset(int axis) {
  return {axis == 0 ? 0.0f : 0.5f, axis == 1 ? 0.0f : 0.5f, axis == 2 ? 0.0f : 0.5f};
}

// Eulerian momentum predictor for physical pressure recovery. Central spatial
// derivatives avoid interpreting semi-Lagrangian interpolation diffusion as a
// physical pressure force. The input includes the slip-wall velocity extension.
__global__ void LfmMomentumPredictor(Grid g, LfmVelocity source, float *output, int axis, float dt) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  Vec3 p = Add(Vec3{float(x), float(y), float(z)}, LfmOffset(axis));
  Vec3 velocity = FaceVelocity(g, source.axis[0], source.axis[1], source.axis[2], p);
  const float *f = source.axis[axis];
  Vec3 gradient{Read(f, g, x + 1, y, z) - Read(f, g, x - 1, y, z), Read(f, g, x, y + 1, z) - Read(f, g, x, y - 1, z),
                Read(f, g, x, y, z + 1) - Read(f, g, x, y, z - 1)};
  output[n] = f[n] - dt * Dot(velocity, gradient) / (2 * g.h);
}

// Quadratic B-spline interpolation and its directional derivative, matching
// the reference LFM N2/dN2 kernels. Maps differentiate the extended velocity
// field rather than a normalized aperture quotient near closed faces.
__device__ float2 LfmQuadratic(float x) {
  float a = fabsf(x);
  if (a < 0.5f)
    return {0.75f - x * x, -2 * x};
  float t = fmaxf(0, 1.5f - a);
  return {0.5f * t * t, -copysignf(t, x)};
}

__device__ float LfmSample(Grid g, const float *field, Vec3 p, int axis, Vec3 tangent, float &derivative) {
  p = Add(p, Scale(LfmOffset(axis), -1));
  int upper[] = {g.nx - (axis != 0), g.ny - (axis != 1), g.nz - (axis != 2)};
  if (p.x < 0 || p.x > upper[0])
    tangent.x = 0;
  if (p.y < 0 || p.y > upper[1])
    tangent.y = 0;
  if (p.z < 0 || p.z > upper[2])
    tangent.z = 0;
  p = {fminf(fmaxf(p.x, 0), float(upper[0])), fminf(fmaxf(p.y, 0), float(upper[1])),
       fminf(fmaxf(p.z, 0), float(upper[2]))};
  int x = int(floorf(p.x - 0.5f)), y = int(floorf(p.y - 0.5f)), z = int(floorf(p.z - 0.5f));
  float2 wx[3], wy[3], wz[3];
  for (int i = 0; i < 3; ++i) {
    wx[i] = LfmQuadratic(p.x - x - i);
    wy[i] = LfmQuadratic(p.y - y - i);
    wz[i] = LfmQuadratic(p.z - z - i);
  }
  float value = 0;
  derivative = 0;
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 3; ++j) {
      // Reuse the transverse products for all three x samples.
      float yz = wy[j].x * wz[k].x;
      float dyz = wy[j].y * wz[k].x * tangent.y + wy[j].x * wz[k].y * tangent.z;
      for (int i = 0; i < 3; ++i) {
        float f = field[g.F(max(0, min(x + i, upper[0])), max(0, min(y + j, upper[1])), max(0, min(z + k, upper[2])))];
        value += f * wx[i].x * yz;
        derivative += f * (wx[i].y * yz * tangent.x + wx[i].x * dyz);
      }
    }
  return value;
}

__device__ Vec3 LfmVelocityAt(Grid g, LfmVelocity velocity, Vec3 p) {
  float unused;
  return {LfmSample(g, velocity.axis[0], p, 0, {}, unused), LfmSample(g, velocity.axis[1], p, 1, {}, unused),
          LfmSample(g, velocity.axis[2], p, 2, {}, unused)};
}

// The reference leapfrog midpoint predictor uses RK2 departure tracing with
// N2 interpolation for both the tracing velocity and the transported component.
__global__ void LfmAdvect(Grid g, LfmVelocity velocity, const float *source, float *destination, int axis, float dt) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  Vec3 p = Add(Vec3{float(x), float(y), float(z)}, LfmOffset(axis));
  Vec3 first = LfmVelocityAt(g, velocity, p);
  Vec3 mid = LfmVelocityAt(g, velocity, Add(p, Scale(first, -0.5f * dt / g.h)));
  float unused;
  destination[n] = LfmSample(g, source, Add(p, Scale(mid, -dt / g.h)), axis, {}, unused);
}

__device__ void LfmDerivative(Grid g, LfmVelocity velocity, Vec3 p, Vec3 tangent, Vec3 &dp, Vec3 &dt) {
  dp.x = LfmSample(g, velocity.axis[0], p, 0, tangent, dt.x);
  dp.y = LfmSample(g, velocity.axis[1], p, 1, tangent, dt.y);
  dp.z = LfmSample(g, velocity.axis[2], p, 2, tangent, dt.z);
  dp = Scale(dp, -1 / g.h);
  dt = Scale(dt, -1 / g.h);
}

__global__ void LfmSampleProbe(Grid g, const float *field, float *result) {
  int n = threadIdx.x;
  if (n >= 4)
    return;
  float y = 6.0f + (n - 1.5f) * 0.02f;
  float derivative;
  result[2 * n] = LfmSample(g, field, {5.3f, y, 6.2f}, 0, {0, 1, 0}, derivative);
  result[2 * n + 1] = derivative;
}

__device__ void LfmRK4(Grid g, LfmVelocity velocity, float dt, Vec3 &p, Vec3 &t) {
  Vec3 p1, p2, p3, p4, t1, t2, t3, t4;
  LfmDerivative(g, velocity, p, t, p1, t1);
  LfmDerivative(g, velocity, Add(p, Scale(p1, dt * 0.5f)), Add(t, Scale(t1, dt * 0.5f)), p2, t2);
  LfmDerivative(g, velocity, Add(p, Scale(p2, dt * 0.5f)), Add(t, Scale(t2, dt * 0.5f)), p3, t3);
  LfmDerivative(g, velocity, Add(p, Scale(p3, dt)), Add(t, Scale(t3, dt)), p4, t4);
  p = Add(p, Scale(Add(Add(p1, Scale(Add(p2, p3), 2)), p4), dt / 6));
  t = Add(t, Scale(Add(Add(t1, Scale(Add(t2, t3), 2)), t4), dt / 6));
}

__global__ void LfmMaps(Grid g, LfmHistory history, float dt, int axis, bool forward, Vec3 *map, Vec3 *jacobian) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  Vec3 p = Add(Vec3{float(x), float(y), float(z)}, LfmOffset(axis));
  Vec3 t{axis == 0 ? 1.0f : 0.0f, axis == 1 ? 1.0f : 0.0f, axis == 2 ? 1.0f : 0.0f};
  for (int i = 0; i < lfm_intervals; ++i)
    LfmRK4(g, history.velocity[forward ? i : lfm_intervals - 1 - i], forward ? -dt : dt, p, t);
  map[n] = p;
  jacobian[n] = t;
}

__global__ void LfmPullback(Grid g,
                            LfmVelocity source,
                            const Vec3 *map,
                            const Vec3 *jacobian,
                            float *destination,
                            const float *subtract) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  Vec3 p = map[n];
  Vec3 value = LfmVelocityAt(g, source, p);
  destination[n] = Dot(jacobian[n], value) - (subtract ? subtract[n] : 0);
}

// Reference LFM limits the corrected impulse against the six adjacent
// uncorrected impulses, after pullback. Initial velocity bounds belong to a
// different covector basis and can incorrectly discard a valid correction.
__global__ void LfmCorrect(Grid g, const float *impulse, float *correction, int axis) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int upper[] = {g.nx - (axis != 0), g.ny - (axis != 1), g.nz - (axis != 2)};
  int p[] = {min(n % (g.nx + 1), upper[0]), min(n / (g.nx + 1) % (g.ny + 1), upper[1]),
             min(n / ((g.nx + 1) * (g.ny + 1)), upper[2])};
  float low = 1e30f, high = -1e30f;
  for (int dimension = 0; dimension < 3; ++dimension)
    for (int sign = -1; sign <= 1; sign += 2) {
      int q[] = {p[0], p[1], p[2]};
      q[dimension] += sign;
      if (q[dimension] < 0 || q[dimension] > upper[dimension])
        continue;
      float value = impulse[g.F(q[0], q[1], q[2])];
      low = fminf(low, value);
      high = fmaxf(high, value);
    }
  correction[n] = fminf(high, fmaxf(low, impulse[n] - 0.5f * correction[n]));
}
