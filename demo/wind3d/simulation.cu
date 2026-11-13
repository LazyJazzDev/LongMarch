#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include "pressure.cuh"
#include "simulation.h"

namespace wind3d {
namespace {
constexpr float density = 1.225f;
constexpr float viscosity = 0.005f;

struct Grid {
  int nx, ny, nz;
  float h;

  __host__ __device__ int C(int x, int y, int z) const {
    return (z * ny + y) * nx + x;
  }

  __host__ __device__ int F(int x, int y, int z) const {
    return (z * (ny + 1) + y) * (nx + 1) + x;
  }

  __device__ bool Inside(int x, int y, int z) const {
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
  }

  __device__ Vec3 Position(int x, int y, int z) const {
    return {(x + 0.5f) * h - 6, (y + 0.5f) * h - 4, (z + 0.5f) * h - 6};
  }
};

__device__ float Read(const float *f, Grid g, int x, int y, int z) {
  return f[g.F(max(0, min(x, g.nx)), max(0, min(y, g.ny)), max(0, min(z, g.nz)))];
}

__device__ float Sample(const float *f, Grid g, Vec3 p) {
  p.x = fminf(fmaxf(p.x, 0), float(g.nx));
  p.y = fminf(fmaxf(p.y, 0), float(g.ny));
  p.z = fminf(fmaxf(p.z, 0), float(g.nz));
  int x = int(p.x), y = int(p.y), z = int(p.z);
  float result = 0;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        result += Read(f, g, x + i, y + j, z + k) * (i ? p.x - x : 1 - p.x + x) * (j ? p.y - y : 1 - p.y + y) *
                  (k ? p.z - z : 1 - p.z + z);
  return result;
}

__global__ void Mask(Grid g, unsigned char *solid, Controls controls) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  solid[n] = AircraftDistance(g.Position(x, y, z), controls) <= 0;
}

__device__ bool Blocked(Grid g, const unsigned char *solid, int x, int y, int z, int axis) {
  if (g.Inside(x, y, z) && solid[g.C(x, y, z)])
    return true;
  if (axis == 0)
    --x;
  if (axis == 1)
    --y;
  if (axis == 2)
    --z;
  return g.Inside(x, y, z) && solid[g.C(x, y, z)];
}

// Subcell sampling of the pressure-cell connection captures thin walls even
// when both adjacent cell centres are fluid. These faces also define A and loads.
__global__ void FaceMask(Grid g,
                         const unsigned char *solid,
                         unsigned char *bu,
                         unsigned char *bv,
                         unsigned char *bw,
                         Controls controls) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  bu[n] = bv[n] = bw[n] = 0;
  Vec3 body = BodyPoint(g.Position(x, y, z), controls);
  if (body.x < -3.1f - g.h || body.x > 3.4f + g.h || fabsf(body.z) > 3.4f + g.h || body.y < -0.85f - g.h ||
      body.y > 1.4f + g.h)
    return;
  unsigned char *faces[] = {bu, bv, bw};
  for (int axis = 0; axis < 3; ++axis) {
    bool blocked = Blocked(g, solid, x, y, z, axis);
    Vec3 normal{axis == 0 ? 1.0f : 0, axis == 1 ? 1.0f : 0, axis == 2 ? 1.0f : 0};
    Vec3 center = Add(g.Position(x, y, z), Scale(normal, -0.5f * g.h));
    for (int sample = -3; sample <= 3 && !blocked; ++sample)
      blocked = AircraftDistance(Add(center, Scale(normal, sample * g.h / 8)), controls) <= 0;
    // A fixed set of samples can pass on either side of a thin trailing edge.
    // Intersect the complete connection with the actual deflected foil solids.
    if (!blocked) {
      Vec3 a = BodyPoint(Add(center, Scale(normal, -0.5f * g.h)), controls);
      Vec3 b = BodyPoint(Add(center, Scale(normal, 0.5f * g.h)), controls);
      for (int side = -1; side <= 1 && !blocked; side += 2)
        for (int part = 0; part < 7 && !blocked; ++part)
          if (part != 5 || side == 1)
            blocked = FoilLineHit(AircraftPart(part, side, controls), a, b);
    }
    faces[axis][n] = blocked;
  }
}

__global__ void Boundary(Grid g,
                         const unsigned char *bu,
                         const unsigned char *bv,
                         const unsigned char *bw,
                         float *u,
                         float *v,
                         float *w,
                         float speed,
                         bool reset) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  if (reset || x == 0 || y == 0 || y == g.ny || z == 0 || z == g.nz) {
    u[n] = speed;
    v[n] = 0;
    w[n] = 0;
  }
  if (bu[n])
    u[n] = 0;
  if (bv[n])
    v[n] = 0;
  if (bw[n])
    w[n] = 0;
}

__device__ Vec3
Departure(Grid g, const float *u, const float *v, const float *w, int x, int y, int z, int axis, float dt) {
  Vec3 offset{axis == 0 ? 0.0f : 0.5f, axis == 1 ? 0.0f : 0.5f, axis == 2 ? 0.0f : 0.5f};
  Vec3 p{x + offset.x, y + offset.y, z + offset.z};
  Vec3 velocity{Sample(u, g, Add(p, {0, -0.5f, -0.5f})), Sample(v, g, Add(p, {-0.5f, 0, -0.5f})),
                Sample(w, g, Add(p, {-0.5f, -0.5f, 0}))};
  Vec3 midpoint = Add(p, Scale(velocity, -0.5f * dt / g.h));
  velocity = {Sample(u, g, Add(midpoint, {0, -0.5f, -0.5f})), Sample(v, g, Add(midpoint, {-0.5f, 0, -0.5f})),
              Sample(w, g, Add(midpoint, {-0.5f, -0.5f, 0}))};
  return Add(Add(p, Scale(velocity, -dt / g.h)), Scale(offset, -1));
}

__global__ void Advect(Grid g,
                       const float *u,
                       const float *v,
                       const float *w,
                       const float *source,
                       float *destination,
                       int axis,
                       float dt) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  destination[n] = Sample(source, g, Departure(g, u, v, w, x, y, z, axis, dt));
}

__global__ void CorrectTransport(Grid g,
                                 const float *u,
                                 const float *v,
                                 const float *w,
                                 const float *source,
                                 const float *reverse,
                                 float *forward,
                                 const unsigned char *bu,
                                 const unsigned char *bv,
                                 const unsigned char *bw,
                                 int axis,
                                 float dt) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  float transported = forward[n];
  if (reverse) {
    Vec3 back = Departure(g, u, v, w, x, y, z, axis, dt);
    int ix = int(floorf(back.x)), iy = int(floorf(back.y)), iz = int(floorf(back.z));
    float low = 1e30f, high = -1e30f;
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          float value = Read(source, g, ix + i, iy + j, iz + k);
          low = fminf(low, value);
          high = fmaxf(high, value);
        }
    transported = fminf(fmaxf(transported + 0.5f * (source[n] - reverse[n]), low), high);
  }
  float lap = 0;
  const unsigned char *blocked[] = {bu, bv, bw};
  for (int direction = 0; direction < 3; ++direction)
    for (int sign = -1; sign <= 1; sign += 2) {
      int neighbour[] = {x, y, z};
      neighbour[direction] += sign;
      float value = Read(source, g, neighbour[0], neighbour[1], neighbour[2]);
      if (direction != axis) {
        // A tangential MAC velocity straddles two pressure cells. Each half
        // of its dual face uses that cell's wall flag. A stationary wall at
        // the midpoint has ghost velocity -u, rather than the fluid velocity
        // on the far side of a thin control surface.
        int face[] = {x, y, z};
        face[direction] += sign > 0;
        float fraction = 0;
        for (int half = 0; half < 2; ++half) {
          if (face[0] >= 0 && face[0] <= g.nx && face[1] >= 0 && face[1] <= g.ny && face[2] >= 0 && face[2] <= g.nz)
            fraction += 0.5f * blocked[direction][g.F(face[0], face[1], face[2])];
          --face[axis];
        }
        value += fraction * (-source[n] - value);
      }
      lap += value - source[n];
    }
  forward[n] = transported + viscosity * dt / (g.h * g.h) * lap;
}

__global__ void Divergence(Grid g,
                           const unsigned char *solid,
                           const float *u,
                           const float *v,
                           const float *w,
                           float *div) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  div[n] = solid[n] ? 0
                    : (u[g.F(x + 1, y, z)] - u[g.F(x, y, z)] + v[g.F(x, y + 1, z)] - v[g.F(x, y, z)] +
                       w[g.F(x, y, z + 1)] - w[g.F(x, y, z)]) /
                          g.h;
}

__global__ void Project(Grid g,
                        const unsigned char *bu,
                        const unsigned char *bv,
                        const unsigned char *bw,
                        const float *p,
                        float *u,
                        float *v,
                        float *w,
                        float dt) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  float *fields[] = {u, v, w};
  const unsigned char *blocked[] = {bu, bv, bw};
  for (int axis = 0; axis < 3; ++axis) {
    int a = x - (axis == 0), b = y - (axis == 1), c = z - (axis == 2);
    if (blocked[axis][n]) {
      fields[axis][n] = 0;
      continue;
    }
    if (g.Inside(x, y, z) && g.Inside(a, b, c))
      fields[axis][n] -= dt / (density * g.h) * (p[g.C(x, y, z)] - p[g.C(a, b, c)]);
    else if (axis == 0 && x == g.nx && y < g.ny && z < g.nz)
      fields[axis][n] += 2 * dt / (density * g.h) * p[g.C(a, b, c)];
  }
}

struct Statistics {
  float before{}, after{}, max_speed{};
  Vec3 force{}, moment{};
  int invalid{};
};

__device__ Vec3 CenterVelocity(Grid g, const float *u, const float *v, const float *w, int x, int y, int z) {
  return {0.5f * (u[g.F(x, y, z)] + u[g.F(x + 1, y, z)]), 0.5f * (v[g.F(x, y, z)] + v[g.F(x, y + 1, z)]),
          0.5f * (w[g.F(x, y, z)] + w[g.F(x, y, z + 1)])};
}

__global__ void Measure(Grid g,
                        const unsigned char *solid,
                        const unsigned char *bu,
                        const unsigned char *bv,
                        const unsigned char *bw,
                        const float *u,
                        const float *v,
                        const float *w,
                        const float *old_u,
                        const float *old_v,
                        const float *old_w,
                        const float *p,
                        const float *before,
                        const float *after,
                        Statistics *blocks) {
  __shared__ Statistics values[256];
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  Statistics value;
  if (n < g.nx * g.ny * g.nz) {
    int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
    Vec3 velocity = CenterVelocity(g, u, v, w, x, y, z);
    value.before = before[n] * before[n];
    value.after = after[n] * after[n];
    value.invalid = !isfinite(velocity.x + velocity.y + velocity.z + p[n] + before[n] + after[n]);
    // Bound every component used by MAC interpolation, including padded faces.
    // Cell-centred averages can cancel even in a divergence-free velocity field.
    int face_count = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    for (int face = n; face < face_count; face += g.nx * g.ny * g.nz) {
      value.max_speed = fmaxf(value.max_speed, fmaxf(fabsf(u[face]), fmaxf(fabsf(v[face]), fabsf(w[face]))));
      value.invalid |= !isfinite(u[face]) || !isfinite(v[face]) || !isfinite(w[face]);
    }
    if (!solid[n]) {
      const unsigned char *blocked[] = {bu, bv, bw};
      for (int axis = 0; axis < 3; ++axis)
        for (int sign = -1; sign <= 1; sign += 2) {
          int a = x + (axis == 0 && sign > 0), b = y + (axis == 1 && sign > 0), c = z + (axis == 2 && sign > 0);
          if (!blocked[axis][g.F(a, b, c)])
            continue;
          Vec3 outward{axis == 0 ? float(sign) : 0, axis == 1 ? float(sign) : 0, axis == 2 ? float(sign) : 0};
          // The explicit viscous update uses the pre-step velocities.
          Vec3 tangent = CenterVelocity(g, old_u, old_v, old_w, x, y, z);
          if (axis == 0)
            tangent.x = 0;
          if (axis == 1)
            tangent.y = 0;
          if (axis == 2)
            tangent.z = 0;
          Vec3 force = Add(Scale(outward, p[n] * g.h * g.h), Scale(tangent, 2 * density * viscosity * g.h));
          Vec3 point = Add(g.Position(x, y, z), Scale(outward, g.h * 0.5f));
          value.force = Add(value.force, force);
          value.moment = Add(value.moment, Cross(point, force));
        }
    }
  }

  values[threadIdx.x] = value;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      auto &a = values[threadIdx.x];
      auto b = values[threadIdx.x + offset];
      a.before += b.before;
      a.after += b.after;
      a.max_speed = fmaxf(a.max_speed, b.max_speed);
      a.force = Add(a.force, b.force);
      a.moment = Add(a.moment, b.moment);
      a.invalid |= b.invalid;
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    blocks[blockIdx.x] = values[0];
}

__global__ void
Export(Grid g, Grid output, const float *u, const float *v, const float *w, const float *p, Cell *cells) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= output.nx * output.ny * output.nz)
    return;
  int x = n % output.nx, y = n / output.nx % output.ny, z = n / (output.nx * output.ny);
  float ratio = float(g.nx) / output.nx;
  Vec3 position{(x + 0.5f) * ratio, (y + 0.5f) * ratio, (z + 0.5f) * ratio};
  Cell cell{Sample(u, g, Add(position, {0, -0.5f, -0.5f})), Sample(v, g, Add(position, {-0.5f, 0, -0.5f})),
            Sample(w, g, Add(position, {-0.5f, -0.5f, 0})), 0};
  position = Add(position, {-0.5f, -0.5f, -0.5f});
  int ix = int(floorf(position.x)), iy = int(floorf(position.y)), iz = int(floorf(position.z));
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        int a = max(0, min(g.nx - 1, ix + i)), b = max(0, min(g.ny - 1, iy + j)), c = max(0, min(g.nz - 1, iz + k));
        cell.pressure += p[g.C(a, b, c)] * (i ? position.x - ix : 1 - position.x + ix) *
                         (j ? position.y - iy : 1 - position.y + iy) * (k ? position.z - iz : 1 - position.z + iz);
      }
  cells[n] = cell;
}
}  // namespace

bool ValidateWallDiffusion(int cuda_device) {
  // A zero-thickness stationary plate separates fluid cells on both sides.
  // Check all six normal/tangent combinations against the analytical shear
  // and against the actual fluid momentum loss from CorrectTransport.
  const Grid g{8, 8, 8, 0.25f};
  constexpr int nf = 9 * 9 * 9, nc = 8 * 8 * 8;
  float *fields = nullptr;
  unsigned char *masks = nullptr;
  Statistics *statistics = nullptr;
  bool passed = cudaSetDevice(cuda_device) == cudaSuccess &&
                cudaMallocManaged(&fields, (4 * nf + nc) * sizeof(float)) == cudaSuccess &&
                cudaMallocManaged(&masks, 3 * nf + nc) == cudaSuccess &&
                cudaMallocManaged(&statistics, 2 * sizeof(Statistics)) == cudaSuccess;
  float dt = 0.1f * g.h * g.h / viscosity;
  for (int normal = 0; passed && normal < 3; ++normal)
    for (int tangent = 0; passed && tangent < 3; ++tangent) {
      if (normal == tangent)
        continue;
      std::fill(fields, fields + 4 * nf + nc, 0.0f);
      std::fill(masks, masks + 3 * nf + nc, 0);
      float *source = fields + tangent * nf, *output = fields + 3 * nf, *zero = fields + 4 * nf;
      std::fill(source, source + nf, 3.0f);
      std::fill(output, output + nf, 3.0f);
      for (int a = 2; a < 6; ++a)
        for (int b = 2; b < 6; ++b) {
          int face[3];
          face[normal] = 4;
          face[(normal + 1) % 3] = a;
          face[(normal + 2) % 3] = b;
          masks[normal * nf + g.F(face[0], face[1], face[2])] = 1;
        }
      CorrectTransport<<<(nf + 255) / 256, 256>>>(g, fields, fields + nf, fields + 2 * nf, source, nullptr, output,
                                                  masks, masks + nf, masks + 2 * nf, tangent, dt);
      Measure<<<2, 256>>>(g, masks + 3 * nf, masks, masks + nf, masks + 2 * nf, fields, fields + nf, fields + 2 * nf,
                          fields, fields + nf, fields + 2 * nf, zero, zero, zero, statistics);
      passed = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
      if (!passed)
        break;
      double momentum_loss = 0;
      for (int i = 0; i < nf; ++i)
        momentum_loss += source[i] - output[i];
      momentum_loss *= density * g.h * g.h * g.h / dt;
      Vec3 force = Add(statistics[0].force, statistics[1].force);
      float measured = tangent == 0 ? force.x : tangent == 1 ? force.y : force.z;
      float expected = 2 * 16 * 2 * density * viscosity * g.h * 3;
      passed = std::abs(measured - expected) < 1e-5f && std::abs(momentum_loss - expected) < 1e-5;
      if (!passed)
        std::cerr << "Wall diffusion mismatch: normal=" << normal << " tangent=" << tangent
                  << " fluid=" << momentum_loss << " wall=" << measured << " expected=" << expected << '\n';
    }
  cudaFree(statistics);
  cudaFree(masks);
  cudaFree(fields);
  std::cout << "wall_diffusion_validation=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

struct Simulation::Impl {
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::thread worker;
  bool stop{}, paused{}, working{};
  uint64_t generation{};
  Controls requested;
  std::shared_ptr<const Snapshot> latest;
  std::string error;
  Grid grid{};
  pressure::Solver pressure_solver;
  SolverOptions options;
  int device{};
  cudaStream_t stream{};
  float *u{}, *v{}, *w{}, *un{}, *vn{}, *wn{}, *reverse{}, *pressure{}, *div{}, *after{};
  unsigned char *solid{}, *bu{}, *bv{}, *bw{};
  Cell *cells{};
  Statistics *statistics{};
  Grid display_grid{};
  std::vector<void *> allocations;

  bool Check(cudaError_t result) {
    if (result == cudaSuccess)
      return true;
    std::lock_guard<std::mutex> guard(mutex);
    error = cudaGetErrorString(result);
    return false;
  }

  template <typename T>
  bool Allocate(T *&p, size_t count) {
    if (!Check(cudaMalloc(reinterpret_cast<void **>(&p), count * sizeof(T))))
      return false;
    allocations.push_back(p);
    return Check(cudaMemsetAsync(p, 0, count * sizeof(T), stream));
  }

  bool Initialize() {
    if (!Check(cudaSetDevice(device)))
      return false;
    int low = 0, high = 0;
    if (!Check(cudaDeviceGetStreamPriorityRange(&low, &high)) ||
        !Check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, low)))
      return false;
    size_t nf = size_t(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1);
    size_t nc = size_t(grid.nx) * grid.ny * grid.nz;
    int display_nx = std::min(grid.nx, 192);
    display_grid = {display_nx, display_nx / 2, display_nx * 3 / 4, 16.0f / display_nx};
    size_t display_cells = size_t(display_grid.nx) * display_grid.ny * display_grid.nz;
    return Allocate(u, nf) && Allocate(v, nf) && Allocate(w, nf) && Allocate(un, nf) && Allocate(vn, nf) &&
           Allocate(wn, nf) && Allocate(reverse, nf) && Allocate(pressure, nc) && Allocate(div, nc) &&
           Allocate(after, nc) && Allocate(solid, nc) && Allocate(bu, nf) && Allocate(bv, nf) && Allocate(bw, nf) &&
           Allocate(cells, display_cells) && Allocate(statistics, (nc + 255) / 256) &&
           Check(pressure_solver.Initialize(grid.nx, grid.ny, grid.nz, stream)) && Check(cudaStreamSynchronize(stream));
  }

  void Run() {
    if (!Check(cudaSetDevice(device)))
      return;
    const auto g = grid;
    int nc = g.nx * g.ny * g.nz, nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    int cb = (nc + 255) / 256, fb = (nf + 255) / 256;
    std::vector<Statistics> measurements(cb);
    size_t display_count = size_t(display_grid.nx) * display_grid.ny * display_grid.nz;
    uint64_t applied_generation = ~uint64_t(0), step = 0;
    float time = 0, max_speed = 12;
    Controls mesh_controls;
    std::shared_ptr<const std::vector<Vertex>> mesh;
    while (true) {
      Controls controls;
      uint64_t current_generation;
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] { return stop || !paused || generation != applied_generation; });
        if (stop)
          break;
        controls = requested;
        current_generation = generation;
        working = true;
      }
      auto start = std::chrono::steady_clock::now();
      bool reset = current_generation != applied_generation;
      if (reset) {
        step = 0;
        time = 0;
        max_speed = controls.speed;
        if (!Check(cudaMemsetAsync(pressure, 0, size_t(nc) * sizeof(float), stream)))
          break;
      }
      applied_generation = current_generation;
      // The advective bound limits displacement to half a cell per component.
      // No-slip ghost values can increase the diffusion diagonal from six
      // to ten at corners; retain a conservative explicit stability bound.
      float dt = std::min(0.5f * g.h / std::max(max_speed, controls.speed), 0.08f * g.h * g.h / viscosity);
      bool geometry_changed = !mesh || controls.angle != mesh_controls.angle || controls.flap != mesh_controls.flap ||
                              controls.aileron != mesh_controls.aileron ||
                              controls.elevator != mesh_controls.elevator || controls.rudder != mesh_controls.rudder;
      if (geometry_changed) {
        Mask<<<cb, 256, 0, stream>>>(g, solid, controls);
        FaceMask<<<fb, 256, 0, stream>>>(g, solid, bu, bv, bw, controls);
        if (!Check(pressure_solver.Build(solid, bu, bv, bw)))
          break;
      }
      Boundary<<<fb, 256, 0, stream>>>(g, bu, bv, bw, u, v, w, controls.speed, reset);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, u, un, 0, dt);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, v, vn, 1, dt);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, w, wn, 2, dt);
      Boundary<<<fb, 256, 0, stream>>>(g, bu, bv, bw, un, vn, wn, controls.speed, false);
      float *sources[] = {u, v, w}, *destinations[] = {un, vn, wn};
      for (int axis = 0; axis < 3; ++axis) {
        if (options.corrected_transport)
          Advect<<<fb, 256, 0, stream>>>(g, u, v, w, destinations[axis], reverse, axis, -dt);
        CorrectTransport<<<fb, 256, 0, stream>>>(g, u, v, w, sources[axis],
                                                 options.corrected_transport ? reverse : nullptr, destinations[axis],
                                                 bu, bv, bw, axis, dt);
      }
      std::swap(u, un);
      std::swap(v, vn);
      std::swap(w, wn);
      Boundary<<<fb, 256, 0, stream>>>(g, bu, bv, bw, u, v, w, controls.speed, false);
      Divergence<<<cb, 256, 0, stream>>>(g, solid, u, v, w, div);
      int pressure_cycles = 0;
      float pressure_residual = 0;
      if (!Check(pressure_solver.Solve(div, pressure, density * g.h * g.h / dt, options.pressure_tolerance,
                                       options.pressure_cycles, pressure_cycles, pressure_residual)))
        break;
      Project<<<fb, 256, 0, stream>>>(g, bu, bv, bw, pressure, u, v, w, dt);
      Divergence<<<cb, 256, 0, stream>>>(g, solid, u, v, w, after);
      Measure<<<cb, 256, 0, stream>>>(g, solid, bu, bv, bw, u, v, w, un, vn, wn, pressure, div, after, statistics);
      Export<<<(display_count + 255) / 256, 256, 0, stream>>>(g, display_grid, u, v, w, pressure, cells);
      auto snapshot = std::make_shared<Snapshot>();
      snapshot->cells.resize(display_count);
      if (!Check(cudaGetLastError()) ||
          !Check(cudaMemcpyAsync(snapshot->cells.data(), cells, display_count * sizeof(Cell), cudaMemcpyDeviceToHost,
                                 stream)) ||
          !Check(cudaMemcpyAsync(measurements.data(), statistics, measurements.size() * sizeof(Statistics),
                                 cudaMemcpyDeviceToHost, stream)) ||
          !Check(cudaStreamSynchronize(stream)))
        break;
      double before_sum = 0, after_sum = 0, force[3]{}, moment[3]{};
      max_speed = controls.speed;
      for (auto value : measurements) {
        before_sum += value.before;
        after_sum += value.after;
        max_speed = std::max(max_speed, value.max_speed);
        if (value.invalid) {
          std::lock_guard<std::mutex> guard(mutex);
          error = "Non-finite fluid state";
          return;
        }
        force[0] += value.force.x;
        force[1] += value.force.y;
        force[2] += value.force.z;
        moment[0] += value.moment.x;
        moment[1] += value.moment.y;
        moment[2] += value.moment.z;
      }
      snapshot->force = {float(force[0]), float(force[1]), float(force[2])};
      snapshot->moment = {float(moment[0]), float(moment[1]), float(moment[2])};
      snapshot->solver_nx = g.nx;
      snapshot->pressure_converged = pressure_residual <= options.pressure_tolerance;
      snapshot->pressure_cycles = pressure_cycles;
      snapshot->pressure_residual = pressure_residual;
      snapshot->nx = display_grid.nx;
      snapshot->ny = display_grid.ny;
      snapshot->nz = display_grid.nz;
      snapshot->step = ++step;
      snapshot->generation = current_generation;
      snapshot->time = time += dt;
      snapshot->dt = dt;
      snapshot->controls = controls;
      if (!mesh || controls.angle != mesh_controls.angle || controls.flap != mesh_controls.flap ||
          controls.aileron != mesh_controls.aileron || controls.elevator != mesh_controls.elevator ||
          controls.rudder != mesh_controls.rudder) {
        mesh = std::make_shared<const std::vector<Vertex>>(AircraftMesh(controls));
        mesh_controls = controls;
      }
      snapshot->aircraft = mesh;
      snapshot->divergence_before = float(std::sqrt(before_sum / nc));
      snapshot->divergence_after = float(std::sqrt(after_sum / nc));
      snapshot->step_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
      {
        std::lock_guard<std::mutex> guard(mutex);
        if (generation == current_generation)
          latest = std::move(snapshot);
        working = false;
      }
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> guard(mutex);
      stop = true;
    }
    wake.notify_one();
    if (worker.joinable())
      worker.join();
    cudaSetDevice(device);
    if (stream)
      cudaStreamSynchronize(stream);
    pressure_solver.Release();
    for (auto allocation : allocations)
      cudaFree(allocation);
    if (stream)
      cudaStreamDestroy(stream);
  }
};

Simulation::Simulation() : impl_(std::make_unique<Impl>()) {
}

Simulation::~Simulation() = default;

std::unique_ptr<Simulation> Simulation::Create(int cuda_device, int nx, const SolverOptions &options) {
  if (cuda_device < 0 || nx < 64 || nx > 640 || nx % 16 || options.pressure_tolerance < 1e-6f ||
      options.pressure_tolerance > 0.1f || options.pressure_cycles < 1 || options.pressure_cycles > 256)
    return {};
  auto result = std::unique_ptr<Simulation>(new Simulation);
  result->impl_->device = cuda_device;
  result->impl_->options = options;
  result->impl_->grid = {nx, nx / 2, nx * 3 / 4, 16.0f / nx};
  if (!result->impl_->Initialize())
    return {};
  result->impl_->worker = std::thread([p = result->impl_.get()] { p->Run(); });
  return result;
}

void Simulation::Configure(const Controls &controls, bool paused) {
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->requested = controls;
    impl_->paused = paused;
  }
  impl_->wake.notify_one();
}

void Simulation::Reset() {
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    ++impl_->generation;
  }
  impl_->wake.notify_one();
}

bool Simulation::Paused() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->paused && !impl_->working;
}

std::shared_ptr<const Snapshot> Simulation::Latest() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->latest;
}

std::string Simulation::Error() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->error;
}
}  // namespace wind3d
