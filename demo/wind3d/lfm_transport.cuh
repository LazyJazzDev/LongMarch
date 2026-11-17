#pragma once

// Included inside simulation.cu's private namespace, after its MAC samplers.
// Sun et al., Leapfrog Flow Maps (SIGGRAPH 2025), sections 4.1-4.3.
// Coordinates and map derivatives here are in grid units. Unlike the reference
// tiled implementation, interpolation uses the wind tunnel's face apertures.
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

// Value and directional derivative of the same aperture-weighted interpolant.
// Differentiating the normalized weights is essential near partial faces.
__device__ float LfmSample(Grid g, const float *field, Vec3 p, int axis, Vec3 tangent, float &derivative) {
  p = Add(p, Scale(LfmOffset(axis), -1));
  if (p.x < 0 || p.x > g.nx)
    tangent.x = 0;
  if (p.y < 0 || p.y > g.ny)
    tangent.y = 0;
  if (p.z < 0 || p.z > g.nz)
    tangent.z = 0;
  p = {fminf(fmaxf(p.x, 0), float(g.nx)), fminf(fmaxf(p.y, 0), float(g.ny)), fminf(fmaxf(p.z, 0), float(g.nz))};
  int x = int(p.x), y = int(p.y), z = int(p.z);
  float value = 0, norm = 0, grad = 0, norm_grad = 0;
  float plain = 0, plain_grad = 0;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        float wx = i ? p.x - x : 1 - p.x + x;
        float wy = j ? p.y - y : 1 - p.y + y;
        float wz = k ? p.z - z : 1 - p.z + z;
        float weight = wx * wy * wz;
        float dw = (i ? 1 : -1) * tangent.x * wy * wz + (j ? 1 : -1) * tangent.y * wx * wz +
                   (k ? 1 : -1) * tangent.z * wx * wy;
        int n = g.F(min(x + i, g.nx), min(y + j, g.ny), min(z + k, g.nz));
        float f = field[n];
        float a = g.aperture[axis] ? g.aperture[axis][n] / 255.0f : 1;
        value += f * weight * a;
        norm += weight * a;
        grad += f * dw * a;
        norm_grad += dw * a;
        plain += f * weight;
        plain_grad += f * dw;
      }
  if (norm < 1e-6f) {
    derivative = plain_grad;
    return plain;
  }
  derivative = (grad * norm - value * norm_grad) / (norm * norm);
  return value / norm;
}

__device__ void LfmDerivative(Grid g, LfmVelocity velocity, Vec3 p, Vec3 tangent, Vec3 &dp, Vec3 &dt) {
  dp.x = LfmSample(g, velocity.axis[0], p, 0, tangent, dt.x);
  dp.y = LfmSample(g, velocity.axis[1], p, 1, tangent, dt.y);
  dp.z = LfmSample(g, velocity.axis[2], p, 2, tangent, dt.z);
  dp = Scale(dp, -1 / g.h);
  dt = Scale(dt, -1 / g.h);
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
  Vec3 value = FaceVelocity(g, source.axis[0], source.axis[1], source.axis[2], p);
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
