#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

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

__global__ void Boundary(Grid g, const unsigned char *solid, float *u, float *v, float *w, float speed, bool reset) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  if (reset || x == 0 || y == 0 || y == g.ny || z == 0 || z == g.nz) {
    u[n] = speed;
    v[n] = 0;
    w[n] = 0;
  }
  if (Blocked(g, solid, x, y, z, 0))
    u[n] = 0;
  if (Blocked(g, solid, x, y, z, 1))
    v[n] = 0;
  if (Blocked(g, solid, x, y, z, 2))
    w[n] = 0;
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
  Vec3 offset{axis == 0 ? 0.0f : 0.5f, axis == 1 ? 0.0f : 0.5f, axis == 2 ? 0.0f : 0.5f};
  Vec3 p{x + offset.x, y + offset.y, z + offset.z};
  Vec3 velocity{Sample(u, g, Add(p, {0, -0.5f, -0.5f})), Sample(v, g, Add(p, {-0.5f, 0, -0.5f})),
                Sample(w, g, Add(p, {-0.5f, -0.5f, 0}))};
  Vec3 back = Add(Add(p, Scale(velocity, -dt / g.h)), Scale(offset, -1));
  float lap = Read(source, g, x - 1, y, z) + Read(source, g, x + 1, y, z) + Read(source, g, x, y - 1, z) +
              Read(source, g, x, y + 1, z) + Read(source, g, x, y, z - 1) + Read(source, g, x, y, z + 1) -
              6 * source[n];
  destination[n] = Sample(source, g, back) + viscosity * dt / (g.h * g.h) * lap;
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

__global__ void Pressure(Grid g, const unsigned char *solid, const float *div, float *p, float dt, int parity) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz || solid[n])
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  if ((x + y + z) % 2 != parity)
    return;
  float sum = 0, diagonal = 0;
  for (int axis = 0; axis < 3; ++axis) {
    for (int sign = -1; sign <= 1; sign += 2) {
      int a = x + (axis == 0 ? sign : 0), b = y + (axis == 1 ? sign : 0), c = z + (axis == 2 ? sign : 0);
      if (g.Inside(a, b, c)) {
        int neighbor = g.C(a, b, c);
        if (!solid[neighbor]) {
          sum += p[neighbor];
          diagonal += 1;
        }
      } else if (a == g.nx) {
        // Outlet pressure is zero at the boundary, half a cell away.
        diagonal += 2;
      }
    }
  }
  if (diagonal > 0) {
    float candidate = (sum - div[n] * density * g.h * g.h / dt) / diagonal;
    p[n] += 1.7f * (candidate - p[n]);
  }
}

__global__ void Project(Grid g, const unsigned char *solid, const float *p, float *u, float *v, float *w, float dt) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  float *fields[] = {u, v, w};
  for (int axis = 0; axis < 3; ++axis) {
    int a = x - (axis == 0), b = y - (axis == 1), c = z - (axis == 2);
    if (Blocked(g, solid, x, y, z, axis)) {
      fields[axis][n] = 0;
      continue;
    }
    if (g.Inside(x, y, z) && g.Inside(a, b, c))
      fields[axis][n] -= dt / (density * g.h) * (p[g.C(x, y, z)] - p[g.C(a, b, c)]);
    else if (axis == 0 && x == g.nx && y < g.ny && z < g.nz)
      fields[axis][n] += 2 * dt / (density * g.h) * p[g.C(a, b, c)];
  }
}

__global__ void Export(Grid g,
                       const unsigned char *solid,
                       const float *u,
                       const float *v,
                       const float *w,
                       const float *p,
                       Cell *cells) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  cells[n] = {0.5f * (u[g.F(x, y, z)] + u[g.F(x + 1, y, z)]), 0.5f * (v[g.F(x, y, z)] + v[g.F(x, y + 1, z)]),
              0.5f * (w[g.F(x, y, z)] + w[g.F(x, y, z + 1)]), solid[n] ? 0 : p[n]};
}
}  // namespace

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
  int device{};
  cudaStream_t stream{};
  float *u{}, *v{}, *w{}, *un{}, *vn{}, *wn{}, *pressure{}, *div{}, *after{};
  unsigned char *solid{};
  Cell *cells{};
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
    return Allocate(u, nf) && Allocate(v, nf) && Allocate(w, nf) && Allocate(un, nf) && Allocate(vn, nf) &&
           Allocate(wn, nf) && Allocate(pressure, nc) && Allocate(div, nc) && Allocate(after, nc) &&
           Allocate(solid, nc) && Allocate(cells, nc) && Check(cudaStreamSynchronize(stream));
  }

  void Run() {
    if (!Check(cudaSetDevice(device)))
      return;
    const auto g = grid;
    int nc = g.nx * g.ny * g.nz, nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    int cb = (nc + 255) / 256, fb = (nf + 255) / 256;
    std::vector<float> before_cpu(nc), after_cpu(nc);
    std::vector<unsigned char> mask(nc);
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
      }
      applied_generation = current_generation;
      float dt = std::min(0.5f * g.h / std::max(max_speed, controls.speed), 0.12f * g.h * g.h / viscosity);
      Mask<<<cb, 256, 0, stream>>>(g, solid, controls);
      Boundary<<<fb, 256, 0, stream>>>(g, solid, u, v, w, controls.speed, reset);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, u, un, 0, dt);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, v, vn, 1, dt);
      Advect<<<fb, 256, 0, stream>>>(g, u, v, w, w, wn, 2, dt);
      std::swap(u, un);
      std::swap(v, vn);
      std::swap(w, wn);
      Boundary<<<fb, 256, 0, stream>>>(g, solid, u, v, w, controls.speed, false);
      Divergence<<<cb, 256, 0, stream>>>(g, solid, u, v, w, div);
      if (!Check(cudaMemsetAsync(pressure, 0, size_t(nc) * sizeof(float), stream)))
        break;
      // Red-black SOR solves the variable solid-boundary Poisson system. Each
      // launch is short so the graphics queue can progress between launches.
      for (int iteration = 0; iteration < 400; ++iteration) {
        Pressure<<<cb, 256, 0, stream>>>(g, solid, div, pressure, dt, 0);
        Pressure<<<cb, 256, 0, stream>>>(g, solid, div, pressure, dt, 1);
      }
      Project<<<fb, 256, 0, stream>>>(g, solid, pressure, u, v, w, dt);
      Divergence<<<cb, 256, 0, stream>>>(g, solid, u, v, w, after);
      Export<<<cb, 256, 0, stream>>>(g, solid, u, v, w, pressure, cells);
      auto snapshot = std::make_shared<Snapshot>();
      snapshot->cells.resize(nc);
      if (!Check(cudaGetLastError()) ||
          !Check(cudaMemcpyAsync(snapshot->cells.data(), cells, size_t(nc) * sizeof(Cell), cudaMemcpyDeviceToHost,
                                 stream)) ||
          !Check(cudaMemcpyAsync(before_cpu.data(), div, size_t(nc) * sizeof(float), cudaMemcpyDeviceToHost, stream)) ||
          !Check(
              cudaMemcpyAsync(after_cpu.data(), after, size_t(nc) * sizeof(float), cudaMemcpyDeviceToHost, stream)) ||
          !Check(cudaMemcpyAsync(mask.data(), solid, size_t(nc), cudaMemcpyDeviceToHost, stream)) ||
          !Check(cudaStreamSynchronize(stream)))
        break;
      double before_sum = 0, after_sum = 0;
      max_speed = controls.speed;
      for (int n = 0; n < nc; ++n) {
        before_sum += double(before_cpu[n]) * before_cpu[n];
        after_sum += double(after_cpu[n]) * after_cpu[n];
        const auto &c = snapshot->cells[n];
        max_speed = std::max(max_speed, Length({c.u, c.v, c.w}));
        if (!std::isfinite(c.u + c.v + c.w + c.pressure + after_cpu[n])) {
          std::lock_guard<std::mutex> guard(mutex);
          error = "Non-finite fluid state";
          return;
        }
        if (!mask[n])
          continue;
        int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
        for (int axis = 0; axis < 3; ++axis)
          for (int sign : {-1, 1}) {
            int a = x + (axis == 0 ? sign : 0), b = y + (axis == 1 ? sign : 0), d = z + (axis == 2 ? sign : 0);
            if (a < 0 || a >= g.nx || b < 0 || b >= g.ny || d < 0 || d >= g.nz)
              continue;
            int neighbor = g.C(a, b, d);
            if (mask[neighbor])
              continue;
            const auto &fluid = snapshot->cells[neighbor];
            Vec3 normal{axis == 0 ? float(sign) : 0, axis == 1 ? float(sign) : 0, axis == 2 ? float(sign) : 0};
            Vec3 tangential{axis == 0 ? 0 : fluid.u, axis == 1 ? 0 : fluid.v, axis == 2 ? 0 : fluid.w};
            Vec3 force =
                Add(Scale(normal, -fluid.pressure * g.h * g.h), Scale(tangential, 2 * density * viscosity * g.h));
            Vec3 point{(x + 0.5f) * g.h - 6 + normal.x * g.h * 0.5f, (y + 0.5f) * g.h - 4 + normal.y * g.h * 0.5f,
                       (z + 0.5f) * g.h - 6 + normal.z * g.h * 0.5f};
            snapshot->force = Add(snapshot->force, force);
            snapshot->moment = Add(snapshot->moment, Cross(point, force));
          }
      }
      snapshot->nx = g.nx;
      snapshot->ny = g.ny;
      snapshot->nz = g.nz;
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
    for (auto allocation : allocations)
      cudaFree(allocation);
    if (stream)
      cudaStreamDestroy(stream);
  }
};

Simulation::Simulation() : impl_(std::make_unique<Impl>()) {
}

Simulation::~Simulation() = default;

std::unique_ptr<Simulation> Simulation::Create(int cuda_device, int nx) {
  if (cuda_device < 0 || nx < 64 || nx > 256 || nx % 16)
    return {};
  auto result = std::unique_ptr<Simulation>(new Simulation);
  result->impl_->device = cuda_device;
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
