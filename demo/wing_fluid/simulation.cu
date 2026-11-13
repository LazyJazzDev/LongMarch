#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "simulation.h"
#include "wing_profile.h"

namespace wing_fluid {
namespace {
constexpr int block = 256;

struct Level {
  int w{}, h{};
  float *p{}, *b{}, *r{}, *east{}, *north{}, *boundary{};
};

__device__ bool Fluid(const unsigned char *mask, int x, int y, int w, int h) {
  return x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x];
}

// An implicit function with the selected profile zero contour (not a distance field).
__device__ float WingImplicit(float x, float y, float angle, int profile) {
  float a = angle * 0.01745329252f;
  x -= 2;
  y -= 1.5f;
  float q = (cosf(a) * x - sinf(a) * y) / Simulation::chord + 0.5f;
  float t = (sinf(a) * x + cosf(a) * y) / Simulation::chord;
  return fmaxf(fabsf(t - WingCamber(q, profile)) - WingThickness(q, profile), fmaxf(-q, q - 1)) * Simulation::chord;
}

__device__ float OpenFraction(float x, float y, float sx, float sy, float angle, int profile) {
  float fraction = 0;
  float a = WingImplicit(x, y, angle, profile);
  // Subsegments also detect the thin leading/trailing edge when both face endpoints are outside.
  for (int k = 1; k <= 8; ++k) {
    float b = WingImplicit(x + sx * (k / 8.0f), y + sy * (k / 8.0f), angle, profile);
    fraction += a >= 0 && b >= 0 ? 1 : a <= 0 && b <= 0 ? 0 : a > 0 ? a / (a - b) : b / (b - a);
    a = b;
  }
  fraction *= 0.125f;
  // Remove vanishing apertures to keep the pressure system well conditioned.
  return fraction < 0.01f ? 0 : fraction > 0.99f ? 1 : fraction;
}

__global__ void SmoothGeometry(Level l, float angle, int profile) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  float h = Simulation::domain_width / l.w;
  int x = i % l.w, y = i / l.w;
  l.east[i] = x + 1 < l.w ? OpenFraction((x + 1) * h, y * h, 0, h, angle, profile) : 0;
  l.north[i] = y + 1 < l.h ? OpenFraction(x * h, (y + 1) * h, h, 0, angle, profile) : 0;
  l.boundary[i] = x == l.w - 1 ? 2 : 0;
}

__global__ void ActiveCells(Level l, unsigned char *mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < l.w * l.h)
    mask[i] =
        l.east[i] + l.north[i] + l.boundary[i] + (i % l.w ? l.east[i - 1] : 1) + (i >= l.w ? l.north[i - l.w] : 0) > 0;
}

__device__ float UWeight(Level l, int x, int y) {
  return x == 0 || x == l.w ? 1 : l.east[y * l.w + x - 1];
}

__device__ float VWeight(Level l, int x, int y) {
  return y == 0 || y == l.h ? 0 : l.north[(y - 1) * l.w + x];
}

__device__ float Flux(Level l, const float *u, const float *v, int x, int y) {
  int j = y * (l.w + 1) + x, i = y * l.w + x;
  return UWeight(l, x + 1, y) * u[j + 1] - UWeight(l, x, y) * u[j] + VWeight(l, x, y + 1) * v[i + l.w] -
         VWeight(l, x, y) * v[i];
}

__global__ void Geometry(unsigned char *mask, int w, int h, float angle, int profile) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  float dx = Simulation::domain_width / w;
  mask[i] = WingImplicit((i % w + 0.5f) * dx, (i / w + 0.5f) * dx, angle, profile) >= 0;
}

__global__ void FineOperator(Level l, const unsigned char *mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  int x = i % l.w, y = i / l.w;
  bool f = mask[i];
  l.east[i] = f && Fluid(mask, x + 1, y, l.w, l.h) ? 1 : 0;
  l.north[i] = f && Fluid(mask, x, y + 1, l.w, l.h) ? 1 : 0;
  // Outflow pressure is zero at the face, half a cell from the last pressure node.
  l.boundary[i] = f && x == l.w - 1 ? 2 : 0;
}

__global__ void CoarsenOperator(Level fine, Level coarse) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= coarse.w * coarse.h)
    return;
  int x = (i % coarse.w) * 2, y = (i / coarse.w) * 2, j = y * fine.w + x;
  coarse.east[i] = 0.5f * (fine.east[j + 1] + fine.east[j + fine.w + 1]);
  coarse.north[i] = 0.5f * (fine.north[j + fine.w] + fine.north[j + fine.w + 1]);
  coarse.boundary[i] =
      0.5f * (fine.boundary[j] + fine.boundary[j + 1] + fine.boundary[j + fine.w] + fine.boundary[j + fine.w + 1]);
}

__device__ float Diagonal(Level l, int i) {
  return l.east[i] + l.north[i] + l.boundary[i] + (i % l.w ? l.east[i - 1] : 0) + (i >= l.w ? l.north[i - l.w] : 0);
}

__device__ float Neighbors(Level l, const float *p, int i) {
  float sum = 0;
  if (i % l.w + 1 < l.w)
    sum += l.east[i] * p[i + 1];
  if (i % l.w)
    sum += l.east[i - 1] * p[i - 1];
  if (i + l.w < l.w * l.h)
    sum += l.north[i] * p[i + l.w];
  if (i >= l.w)
    sum += l.north[i - l.w] * p[i - l.w];
  return sum;
}

__global__ void Smooth(Level l, int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h || ((i % l.w + i / l.w) & 1) != color)
    return;
  float d = Diagonal(l, i);
  l.p[i] = d > 0 ? (l.b[i] + Neighbors(l, l.p, i)) / d : 0;
}

__global__ void Residual(Level l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < l.w * l.h)
    l.r[i] = l.b[i] - (Diagonal(l, i) * l.p[i] - Neighbors(l, l.p, i));
}

// The last grid is 4 x 2. Solve it directly instead of launching many tiny smoothing kernels.
__global__ void CoarseSolve(Level l) {
  float matrix[8][9]{};
  for (int i = 0; i < 8; ++i) {
    float d = Diagonal(l, i);
    matrix[i][i] = d > 0 ? d : 1;
    matrix[i][8] = d > 0 ? l.b[i] : 0;
    if (i % l.w + 1 < l.w)
      matrix[i][i + 1] = -l.east[i];
    if (i % l.w)
      matrix[i][i - 1] = -l.east[i - 1];
    if (i + l.w < 8)
      matrix[i][i + l.w] = -l.north[i];
    if (i >= l.w)
      matrix[i][i - l.w] = -l.north[i - l.w];
  }
  for (int k = 0; k < 8; ++k) {
    for (int i = k + 1; i < 8; ++i) {
      float scale = matrix[i][k] / matrix[k][k];
      for (int j = k + 1; j <= 8; ++j)
        matrix[i][j] -= scale * matrix[k][j];
    }
  }
  for (int i = 7; i >= 0; --i) {
    float value = matrix[i][8];
    for (int j = i + 1; j < 8; ++j)
      value -= matrix[i][j] * l.p[j];
    l.p[i] = value / matrix[i][i];
  }
}

__global__ void Restrict(Level f, Level c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c.w * c.h)
    return;
  int j = (i / c.w * 2) * f.w + (i % c.w * 2);
  c.b[i] = f.r[j] + f.r[j + 1] + f.r[j + f.w] + f.r[j + f.w + 1];
  c.p[i] = 0;
}

__device__ float CoarsePressure(Level c, int x, int y) {
  float sign = x >= c.w ? -1.0f : 1.0f;
  return sign * c.p[min(max(y, 0), c.h - 1) * c.w + min(max(x, 0), c.w - 1)];
}

__global__ void Prolong(Level c, Level f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.w * f.h || Diagonal(f, i) == 0)
    return;
  float x = (i % f.w + 0.5f) * 0.5f - 0.5f, y = (i / f.w + 0.5f) * 0.5f - 0.5f;
  int ix = int(floorf(x)), iy = int(floorf(y));
  float a = x - ix, b = y - iy;
  f.p[i] += (1 - b) * ((1 - a) * CoarsePressure(c, ix, iy) + a * CoarsePressure(c, ix + 1, iy)) +
            b * ((1 - a) * CoarsePressure(c, ix, iy + 1) + a * CoarsePressure(c, ix + 1, iy + 1));
}

__device__ float Sample(const float *f, int w, int h, float x, float y) {
  x = fminf(fmaxf(x, 0), float(w - 1));
  y = fminf(fmaxf(y, 0), float(h - 1));
  int ix = int(x), iy = int(y), jx = min(ix + 1, w - 1), jy = min(iy + 1, h - 1);
  float a = x - ix, b = y - iy;
  return (1 - b) * ((1 - a) * f[iy * w + ix] + a * f[iy * w + jx]) +
         b * ((1 - a) * f[jy * w + ix] + a * f[jy * w + jx]);
}

__device__ float2 Velocity(const float *u, const float *v, int w, int h, float x, float y) {
  return make_float2(Sample(u, w + 1, h, x, y - 0.5f), Sample(v, w, h + 1, x - 0.5f, y));
}

__global__ void InitializeVelocity(float *u, float *v, int w, int h, float speed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (w + 1) * h)
    u[i] = speed;
  if (i < w * (h + 1))
    v[i] = 0;
}

__global__ void InitializeDye(float *dye, const unsigned char *mask, int w, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  float stripe = sinf((i / w + 0.5f) * (Simulation::domain_width / w) * 3.14159265f / 0.12f);
  dye[i] = mask[i] ? 0.75f * expf(-50 * stripe * stripe) : 0;
}

__global__ void Boundaries(float *u, float *v, Level l, float speed) {
  int w = l.w, h = l.h;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (w + 1) * h) {
    int x = i % (w + 1), y = i / (w + 1);
    if (x == 0)
      u[i] = speed;
    else if (UWeight(l, x, y) == 0)
      u[i] = 0;
  }
  if (i < w * (h + 1)) {
    int x = i % w, y = i / w;
    if (VWeight(l, x, y) == 0)
      v[i] = 0;
  }
}

__global__ void Advect(const float *u, const float *v, float *next_u, float *next_v, int w, int h, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float dx = Simulation::domain_width / w, tau = dt / dx;
  for (int component = 0; component < 2; ++component) {
    int fw = component ? w : w + 1, fh = component ? h + 1 : h;
    if (i >= fw * fh)
      continue;
    int x = i % fw, y = i / fw;
    float px = x + (component ? 0.5f : 0), py = y + (component ? 0 : 0.5f);
    float2 vel = Velocity(u, v, w, h, px, py);
    float2 mid = Velocity(u, v, w, h, px - 0.5f * tau * vel.x, py - 0.5f * tau * vel.y);
    const float *f = component ? v : u;
    float value = Sample(f, fw, fh, x - tau * mid.x, y - tau * mid.y);
    (component ? next_v : next_u)[i] = value;
  }
}

__global__ void Diffuse(const float *u, const float *v, float *next_u, float *next_v, Level l, float factor) {
  int w = l.w, h = l.h;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int component = 0; component < 2; ++component) {
    int fw = component ? w : w + 1, fh = component ? h + 1 : h;
    if (i >= fw * fh)
      continue;
    int x = i % fw, y = i / fw;
    const float *f = component ? v : u;
    float lap = f[y * fw + max(x - 1, 0)] + f[y * fw + min(x + 1, fw - 1)] + f[max(y - 1, 0) * fw + x] +
                f[min(y + 1, fh - 1) * fw + x] - 4 * f[i];
    float aperture = component ? VWeight(l, x, y) : UWeight(l, x, y);
    // Approximate no-slip drag on the blocked portion, implicit to avoid a tiny-aperture timestep.
    float drag = 2 * factor * (1 - aperture) / fmaxf(aperture, 0.1f);
    (component ? next_v : next_u)[i] = (f[i] + factor * lap) / (1 + drag);
  }
}

__global__ void RHS(Level l, const float *u, const float *v, const unsigned char *mask, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  int x = i % l.w, y = i / l.w;
  float flux = Flux(l, u, v, x, y);
  l.b[i] = mask[i] ? -flux * (Simulation::domain_width / l.w) / dt : 0;
  l.p[i] = 0;
}

__global__ void Project(Level l, float *u, float *v, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float scale = dt / (Simulation::domain_width / l.w);
  if (i < (l.w + 1) * l.h) {
    int x = i % (l.w + 1), y = i / (l.w + 1), j = y * l.w + x;
    if (x == l.w)
      u[i] += scale * 2 * l.p[j - 1];
    else if (x > 0 && l.east[j - 1] > 0)
      u[i] -= scale * (l.p[j] - l.p[j - 1]);
  }
  if (i < l.w * (l.h + 1)) {
    int y = i / l.w;
    if (y > 0 && y < l.h && l.north[i - l.w] > 0)
      v[i] -= scale * (l.p[i] - l.p[i - l.w]);
  }
}

__global__ void AdvectDye(const float *u, const float *v, const float *dye, float *next, int w, int h, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float dx = Simulation::domain_width / w;
  float2 vel = Velocity(u, v, w, h, x + 0.5f, y + 0.5f);
  next[i] = Sample(dye, w, h, x - dt / dx * vel.x, y - dt / dx * vel.y);
}

__global__ void CorrectDye(const float *u,
                           const float *v,
                           const float *dye,
                           const float *forward,
                           float *next,
                           const unsigned char *mask,
                           int w,
                           int h,
                           float dt,
                           float time) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float dx = Simulation::domain_width / w;
  float2 vel = Velocity(u, v, w, h, x + 0.5f, y + 0.5f);
  float value = forward[i] + 0.5f * (dye[i] - Sample(forward, w, h, x + dt / dx * vel.x, y + dt / dx * vel.y));
  int sx = max(0, min(w - 2, int(floorf(x - dt / dx * vel.x))));
  int sy = max(0, min(h - 2, int(floorf(y - dt / dx * vel.y))));
  float lo = 1, hi = 0;
  for (int yy = 0; yy < 2; ++yy)
    for (int xx = 0; xx < 2; ++xx) {
      float d = dye[(sy + yy) * w + sx + xx];
      lo = fminf(lo, d);
      hi = fmaxf(hi, d);
    }
  value = fminf(hi, fmaxf(lo, value)) * expf(-dt * 0.015f);
  if (x < 3) {
    float stripe = sinf((y + 0.5f) * dx * 3.14159265f / 0.12f);
    value = expf(-50 * stripe * stripe) * (0.75f + 0.25f * sinf(time * 5));
  }
  next[i] = mask[i] ? value : 0;
}

__global__ void Display(float4 *out,
                        const float *u,
                        const float *v,
                        Level l,
                        const float *dye,
                        const unsigned char *mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  float2 vel = Velocity(u, v, l.w, l.h, i % l.w + 0.5f, i / l.w + 0.5f);
  out[i] = make_float4(vel.x, vel.y, l.p[i], mask[i] ? dye[i] : -1);
}

__global__ void MeasureFields(Level l,
                              const float *u,
                              const float *v,
                              const unsigned char *mask,
                              float dt,
                              float *stats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  int x = i % l.w, y = i / l.w;
  float dx = Simulation::domain_width / l.w;
  float div = Flux(l, u, v, x, y) / dx;
  float2 vel = Velocity(u, v, l.w, l.h, x + 0.5f, y + 0.5f);
  if (!isfinite(div) || !isfinite(l.p[i]) || !isfinite(vel.x) || !isfinite(vel.y))
    atomicAdd(stats + 6, 1.0f);
  if (mask[i]) {
    float before = l.b[i] * dt / (dx * dx);
    atomicAdd(stats, before * before);
    atomicAdd(stats + 1, div * div);
    atomicAdd(stats + 2, l.r[i] * l.r[i]);
    atomicAdd(stats + 3, 1.0f);
    atomicMax(reinterpret_cast<int *>(stats + 4), __float_as_int(hypotf(vel.x, vel.y)));
  } else {
    atomicMax(reinterpret_cast<int *>(stats + 5), __float_as_int(hypotf(vel.x, vel.y)));
  }
}

__global__ void Manufactured(Level l, float *exact, bool rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h)
    return;
  if (!rhs)
    exact[i] = cosf(1.7f * (i % l.w + 0.5f) / l.w) * sinf(3.1f * (i / l.w + 0.5f) / l.h);
  else {
    l.b[i] = Diagonal(l, i) * exact[i] - Neighbors(l, exact, i);
    l.p[i] = 0;
  }
}

constexpr int surface_segments = 512;

// Counterclockwise closed polygon, cosine spacing resolves the rounded nose.
__host__ __device__ float2 SurfacePoint(int index, float angle, int profile) {
  int k = index <= surface_segments / 2 ? index : surface_segments - index;
  float q = 0.5f * (1 - cosf(k * (6.28318530718f / surface_segments)));
  float x = (q - 0.5f) * Simulation::chord;
  float y = (WingCamber(q, profile) + WingThickness(q, profile) * (index <= surface_segments / 2 ? -1 : 1)) *
            Simulation::chord;
  float a = angle * 0.01745329252f;
  return make_float2(x * cosf(a) + y * sinf(a), -x * sinf(a) + y * cosf(a));
}

__device__ float SurfacePressure(Level l, const unsigned char *mask, float x, float y) {
  int ix = int(floorf(x)), iy = int(floorf(y));
  float a = x - ix, b = y - iy, sum = 0, weight = 0;
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      if (!Fluid(mask, ix + dx, iy + dy, l.w, l.h))
        continue;
      float w = (dx ? a : 1 - a) * (dy ? b : 1 - b);
      sum += w * l.p[(iy + dy) * l.w + ix + dx];
      weight += w;
    }
  return weight > 0 ? sum / weight : 0;
}

__global__ void IntegrateSmoothLoads(Level l,
                                     const float *u,
                                     const float *v,
                                     const unsigned char *mask,
                                     float angle,
                                     int profile,
                                     float viscosity,
                                     double *sums,
                                     int analytic) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= surface_segments)
    return;
  float2 a = SurfacePoint(i, angle, profile), b = SurfacePoint(i + 1, angle, profile);
  double rx = 0.5 * (a.x + b.x), ry = 0.5 * (a.y + b.y);
  double tx = b.x - a.x, ty = b.y - a.y, length = hypot(tx, ty);
  double nx = ty / length, ny = -tx / length;
  float h = Simulation::domain_width / l.w;
  float px = float((rx + 2 + nx * h) / h), py = float((ry + 1.5 + ny * h) / h);
  double p = analytic == 1 ? 7 : analytic == 2 ? 0.7 * rx - 0.4 * ry : SurfacePressure(l, mask, px - 0.5f, py - 0.5f);
  double fx = -p * ty, fy = p * tx;
  float2 vel = Velocity(u, v, l.w, l.h, px, py);
  double normal = vel.x * nx + vel.y * ny;
  double sx = analytic ? 0 : viscosity * (vel.x - normal * nx) * length / h;
  double sy = analytic ? 0 : viscosity * (vel.y - normal * ny) * length / h;
  atomicAdd(sums, fx);
  atomicAdd(sums + 1, fy);
  atomicAdd(sums + 2, rx * fy - ry * fx);
  atomicAdd(sums + 3, sx);
  atomicAdd(sums + 4, sy);
  atomicAdd(sums + 5, rx * sy - ry * sx);
}

__global__ void IntegrateLoads(Level l,
                               const float *u,
                               const float *v,
                               const unsigned char *mask,
                               float viscosity,
                               double *sums,
                               int analytic = 0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= l.w * l.h || !mask[i])
    return;
  int x = i % l.w, y = i / l.w;
  const int ox[4] = {1, -1, 0, 0}, oy[4] = {0, 0, 1, -1};
  const double h = double(Simulation::domain_width) / l.w;
  float2 velocity = Velocity(u, v, l.w, l.h, x + 0.5f, y + 0.5f);
  double local[6]{};
  bool boundary = false;
  for (int k = 0; k < 4; ++k) {
    int nx = x + ox[k], ny = y + oy[k];
    if (nx < 0 || nx >= l.w || ny < 0 || ny >= l.h || mask[ny * l.w + nx])
      continue;
    boundary = true;
    // d points from fluid to solid, opposite to the body's outward normal.
    double rx = (x + 0.5 + 0.5 * ox[k]) * h - 2.0;
    double ry = (y + 0.5 + 0.5 * oy[k]) * h - 1.5;
    double p = analytic == 1 ? 7.0 : analytic == 2 ? 0.7 * rx - 0.4 * ry : l.p[i];
    double fx = p * ox[k] * h, fy = p * oy[k] * h;
    local[0] += fx;
    local[1] += fy;
    local[2] += rx * fy - ry * fx;
    // Stationary no-slip wall: tangential shear from the adjacent cell center,
    // half a cell from the wall. Normal viscous traction is zero for a smooth
    // incompressible no-slip wall; stair-step corners remain an approximation.
    double normal = velocity.x * ox[k] + velocity.y * oy[k];
    double sx = analytic ? 0 : 2 * viscosity * (velocity.x - normal * ox[k]);
    double sy = analytic ? 0 : 2 * viscosity * (velocity.y - normal * oy[k]);
    local[3] += sx;
    local[4] += sy;
    local[5] += rx * sy - ry * sx;
  }
  if (boundary)
    for (int k = 0; k < 6; ++k)
      atomicAdd(sums + k, local[k]);
}
}  // namespace

struct Simulation::Impl {
  int w{}, h{};
  cudaStream_t stream{};
  cudaGraphExec_t pressure_graph{};
  int graph_cycles{};
  std::vector<Level> levels;
  std::vector<void *> allocations;
  unsigned char *mask{};
  float *u{}, *v{}, *next_u{}, *next_v{}, *dye{}, *next_dye{}, *corrected_dye{}, *stats{};
  double *load_sums{};
  Parameters parameters;
  bool loads_valid{};
  float angle{1000}, time{}, last_dt{0.001f};
  bool smooth_boundary{};
  int profile{};
  std::string error;

  bool Check(cudaError_t result) {
    if (result == cudaSuccess)
      return true;
    error = cudaGetErrorString(result);
    return false;
  }

  template <typename T>
  bool Allocate(T *&ptr, size_t count) {
    if (!Check(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T))))
      return false;
    allocations.push_back(ptr);
    return Check(cudaMemsetAsync(ptr, 0, count * sizeof(T), stream));
  }

  int Blocks(int n) const {
    return (n + block - 1) / block;
  }

  bool ReadLoads(double *values, int analytic = 0) {
    if (!Check(cudaMemsetAsync(load_sums, 0, 6 * sizeof(double), stream)))
      return false;
    if (smooth_boundary)
      IntegrateSmoothLoads<<<Blocks(surface_segments), block, 0, stream>>>(levels[0], u, v, mask, angle, profile,
                                                                           parameters.viscosity, load_sums, analytic);
    else
      IntegrateLoads<<<Blocks(w * h), block, 0, stream>>>(levels[0], u, v, mask, parameters.viscosity, load_sums,
                                                          analytic);
    return Check(cudaMemcpyAsync(values, load_sums, 6 * sizeof(double), cudaMemcpyDeviceToHost, stream)) &&
           Check(cudaStreamSynchronize(stream));
  }

  void GeometryFor(const Parameters &p) {
    angle = p.angle;
    profile = p.profile;
    smooth_boundary = p.smooth_boundary;
    if (smooth_boundary) {
      SmoothGeometry<<<Blocks(w * h), block, 0, stream>>>(levels[0], angle, profile);
      ActiveCells<<<Blocks(w * h), block, 0, stream>>>(levels[0], mask);
    } else {
      Geometry<<<Blocks(w * h), block, 0, stream>>>(mask, w, h, angle, profile);
      FineOperator<<<Blocks(w * h), block, 0, stream>>>(levels[0], mask);
    }
    for (size_t i = 1; i < levels.size(); ++i)
      CoarsenOperator<<<Blocks(levels[i].w * levels[i].h), block, 0, stream>>>(levels[i - 1], levels[i]);
  }

  void Relax(Level l, int sweeps) {
    for (int i = 0; i < sweeps; ++i)
      for (int color = 0; color < 2; ++color)
        Smooth<<<Blocks(l.w * l.h), block, 0, stream>>>(l, color);
  }

  void Cycle(size_t index) {
    Level l = levels[index];
    if (index + 1 == levels.size()) {
      CoarseSolve<<<1, 1, 0, stream>>>(l);
      return;
    }
    Relax(l, 3);
    Residual<<<Blocks(l.w * l.h), block, 0, stream>>>(l);
    Level c = levels[index + 1];
    Restrict<<<Blocks(c.w * c.h), block, 0, stream>>>(l, c);
    Cycle(index + 1);
    Prolong<<<Blocks(l.w * l.h), block, 0, stream>>>(c, l);
    Relax(l, 3);
  }

  bool Solve(int cycles) {
    if (!pressure_graph || graph_cycles != cycles) {
      if (!Check(cudaStreamSynchronize(stream)))
        return false;
      if (pressure_graph) {
        cudaGraphExecDestroy(pressure_graph);
        pressure_graph = nullptr;
      }
      if (!Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)))
        return false;
      for (int i = 0; i < cycles; ++i)
        Cycle(0);
      cudaGraph_t graph{};
      if (!Check(cudaStreamEndCapture(stream, &graph)))
        return false;
      auto result = cudaGraphInstantiate(&pressure_graph, graph, nullptr, nullptr, 0);
      cudaGraphDestroy(graph);
      if (!Check(result))
        return false;
      graph_cycles = cycles;
    }
    return Check(cudaGraphLaunch(pressure_graph, stream));
  }
};

Simulation::Simulation() : impl_(std::make_unique<Impl>()) {
}

Simulation::~Simulation() {
  if (!impl_->allocations.empty())
    cudaStreamSynchronize(impl_->stream);
  if (impl_->pressure_graph)
    cudaGraphExecDestroy(impl_->pressure_graph);
  for (void *p : impl_->allocations)
    cudaFree(p);
}

bool Simulation::Initialize(int width, int height, cudaStream_t stream) {
  auto &s = *impl_;
  if (width < 64 || height * 2 != width || (width & (width - 1)) || !s.allocations.empty()) {
    s.error = "Grid must be a power-of-two 2:1 rectangle, initialized once";
    return false;
  }
  s.w = width;
  s.h = height;
  s.stream = stream;
  int w = width, h = height;
  while (true) {
    Level l;
    l.w = w;
    l.h = h;
    if (!s.Allocate(l.p, w * h) || !s.Allocate(l.b, w * h) || !s.Allocate(l.r, w * h) || !s.Allocate(l.east, w * h) ||
        !s.Allocate(l.north, w * h) || !s.Allocate(l.boundary, w * h))
      return false;
    s.levels.push_back(l);
    if (w <= 4)
      break;
    w /= 2;
    h /= 2;
  }
  return s.Allocate(s.mask, width * height) && s.Allocate(s.u, (width + 1) * height) &&
         s.Allocate(s.next_u, (width + 1) * height) && s.Allocate(s.v, width * (height + 1)) &&
         s.Allocate(s.next_v, width * (height + 1)) && s.Allocate(s.dye, width * height) &&
         s.Allocate(s.next_dye, width * height) && s.Allocate(s.corrected_dye, width * height) &&
         s.Allocate(s.stats, 8) && s.Allocate(s.load_sums, 6);
}

bool Simulation::Reset(const Parameters &parameters) {
  auto &s = *impl_;
  s.time = 0;
  s.parameters = parameters;
  s.loads_valid = false;
  s.GeometryFor(parameters);
  InitializeVelocity<<<s.Blocks(s.w * (s.h + 1)), block, 0, s.stream>>>(s.u, s.v, s.w, s.h, parameters.speed);
  Boundaries<<<s.Blocks(s.w * (s.h + 1)), block, 0, s.stream>>>(s.u, s.v, s.levels[0], parameters.speed);
  for (auto l : s.levels)
    if (!s.Check(cudaMemsetAsync(l.p, 0, l.w * l.h * sizeof(float), s.stream)))
      return false;
  InitializeDye<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.dye, s.mask, s.w, s.h);
  // A reset is also a valid incompressible state, including when the UI is paused.
  // Unit dt projects the initial velocity without advancing simulation time.
  s.last_dt = 1;
  RHS<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.levels[0], s.u, s.v, s.mask, s.last_dt);
  for (int i = 0; i < 32; ++i)
    s.Cycle(0);
  Project<<<s.Blocks(s.w * (s.h + 1)), block, 0, s.stream>>>(s.levels[0], s.u, s.v, s.last_dt);
  return s.Check(cudaGetLastError());
}

bool Simulation::Step(const Parameters &parameters, float dt) {
  auto &s = *impl_;
  s.parameters = parameters;
  s.loads_valid = true;
  s.last_dt = dt;
  if (parameters.angle != s.angle || parameters.smooth_boundary != s.smooth_boundary || parameters.profile != s.profile)
    s.GeometryFor(parameters);
  int faces = s.Blocks(s.w * (s.h + 1));
  Boundaries<<<faces, block, 0, s.stream>>>(s.u, s.v, s.levels[0], parameters.speed);
  Advect<<<faces, block, 0, s.stream>>>(s.u, s.v, s.next_u, s.next_v, s.w, s.h, dt);
  Boundaries<<<faces, block, 0, s.stream>>>(s.next_u, s.next_v, s.levels[0], parameters.speed);
  float dx = Simulation::domain_width / s.w;
  Diffuse<<<faces, block, 0, s.stream>>>(s.next_u, s.next_v, s.u, s.v, s.levels[0],
                                         parameters.viscosity * dt / (dx * dx));
  Boundaries<<<faces, block, 0, s.stream>>>(s.u, s.v, s.levels[0], parameters.speed);
  RHS<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.levels[0], s.u, s.v, s.mask, dt);
  if (!s.Solve(parameters.cycles))
    return false;
  Project<<<faces, block, 0, s.stream>>>(s.levels[0], s.u, s.v, dt);
  s.time += dt;
  AdvectDye<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.u, s.v, s.dye, s.next_dye, s.w, s.h, dt);
  CorrectDye<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.u, s.v, s.dye, s.next_dye, s.corrected_dye, s.mask, s.w,
                                                          s.h, dt, s.time);
  std::swap(s.dye, s.corrected_dye);
  return s.Check(cudaGetLastError());
}

bool Simulation::Export(void *buffer) {
  auto &s = *impl_;
  Display<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(static_cast<float4 *>(buffer), s.u, s.v, s.levels[0], s.dye,
                                                       s.mask);
  return s.Check(cudaGetLastError());
}

bool Simulation::Measure(Diagnostics &d) {
  auto &s = *impl_;
  if (!s.Check(cudaMemsetAsync(s.stats, 0, 8 * sizeof(float), s.stream)))
    return false;
  Residual<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.levels[0]);
  MeasureFields<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(s.levels[0], s.u, s.v, s.mask, s.last_dt, s.stats);
  float values[8];
  if (!s.Check(cudaMemcpyAsync(values, s.stats, sizeof(values), cudaMemcpyDeviceToHost, s.stream)) ||
      !s.Check(cudaStreamSynchronize(s.stream)))
    return false;
  float count = std::max(1.0f, values[3]);
  d = {std::sqrt(values[0] / count),
       std::sqrt(values[1] / count),
       std::sqrt(values[2] / count),
       values[4],
       values[5],
       values[6] == 0};
  return true;
}

bool Simulation::MeasureLoads(LoadCoefficients &loads) {
  auto &s = *impl_;
  loads = {};
  loads.angle = s.angle;
  loads.profile = s.profile;
  if (!s.loads_valid)
    return true;
  double values[6];
  if (!s.ReadLoads(values))
    return false;
  double reference = 0.5 * s.parameters.speed * s.parameters.speed * chord;
  loads.x = float((values[0] + values[3]) / reference);
  loads.y = float((values[1] + values[4]) / reference);
  loads.moment = float((values[2] + values[5]) / (reference * chord));
  loads.valid = std::isfinite(loads.x) && std::isfinite(loads.y) && std::isfinite(loads.moment);
  if (!loads.valid)
    s.error = "Nonfinite surface load";
  return loads.valid;
}

bool Simulation::ValidateLoads(float &relative_error) {
  auto &s = *impl_;
  std::vector<unsigned char> mask(s.w * s.h);
  if (!s.Check(cudaMemcpyAsync(mask.data(), s.mask, mask.size(), cudaMemcpyDeviceToHost, s.stream)) ||
      !s.Check(cudaStreamSynchronize(s.stream)))
    return false;
  double area = 0, mx = 0, my = 0, h = double(domain_width) / s.w;
  for (int i = 0; i < s.w * s.h; ++i)
    if (!mask[i]) {
      area += h * h;
      mx += h * h * ((i % s.w + 0.5) * h - 2);
      my += h * h * ((i / s.w + 0.5) * h - 1.5);
    }
  double constant[6], linear[6];
  if (s.smooth_boundary) {
    area = mx = my = 0;
    // Polygon area and first moments, independent of the surface traction integral.
    for (int i = 0; i < surface_segments; ++i) {
      auto a = SurfacePoint(i, s.angle, s.profile), b = SurfacePoint(i + 1, s.angle, s.profile);
      double cross = double(a.x) * b.y - double(b.x) * a.y;
      area += cross / 2;
      mx += (a.x + b.x) * cross / 6;
      my += (a.y + b.y) * cross / 6;
    }
  }
  if (!s.ReadLoads(constant, 1) || !s.ReadLoads(linear, 2))
    return false;
  // Divergence theorem on the closed voxel body, independent of its perimeter.
  // Constant pressure gives zero force/torque; p = gx*x + gy*y gives -area*grad(p).
  double expected[3] = {-0.7 * area, 0.4 * area, 0.4 * mx + 0.7 * my};
  double error = 0;
  for (int k = 0; k < 3; ++k) {
    error = std::max(error, std::abs(constant[k]));
    error = std::max(error, std::abs(linear[k] - expected[k]));
  }
  relative_error = float(error / std::max(area, 1e-20));
  return true;
}

bool Simulation::ValidatePressure(float &residual, float &error) {
  auto &s = *impl_;
  Level l = s.levels[0];
  // Reuse the idle dye scratch as an exact solution; b is the discrete A applied to it.
  Manufactured<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(l, s.next_dye, false);
  Manufactured<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(l, s.next_dye, true);
  for (int i = 0; i < 40; ++i)
    s.Cycle(0);
  Residual<<<s.Blocks(s.w * s.h), block, 0, s.stream>>>(l);
  std::vector<float> b(s.w * s.h), r(b.size()), p(b.size()), exact(b.size());
  std::vector<unsigned char> mask(b.size());
  for (auto pair : {std::make_pair(b.data(), l.b), std::make_pair(r.data(), l.r), std::make_pair(p.data(), l.p),
                    std::make_pair(exact.data(), s.next_dye)})
    if (!s.Check(cudaMemcpyAsync(pair.first, pair.second, b.size() * sizeof(float), cudaMemcpyDeviceToHost, s.stream)))
      return false;
  if (!s.Check(cudaMemcpyAsync(mask.data(), s.mask, mask.size(), cudaMemcpyDeviceToHost, s.stream)))
    return false;
  if (!s.Check(cudaStreamSynchronize(s.stream)))
    return false;
  double b2 = 0, r2 = 0, e2 = 0, p2 = 0;
  for (size_t i = 0; i < b.size(); ++i) {
    b2 += double(b[i]) * b[i];
    r2 += double(r[i]) * r[i];
    if (mask[i]) {
      e2 += double(p[i] - exact[i]) * (p[i] - exact[i]);
      p2 += double(exact[i]) * exact[i];
    }
  }
  residual = float(std::sqrt(r2 / std::max(b2, 1e-30)));
  error = float(std::sqrt(e2 / std::max(p2, 1e-30)));
  return true;
}

int Simulation::Levels() const {
  return int(impl_->levels.size());
}

const std::string &Simulation::Error() const {
  return impl_->error;
}
}  // namespace wing_fluid
