#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <vector>

#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"
#include "amr_gpu_transport_kernels.cuh"
#include "wall_model.h"

namespace wind3d::amr {
namespace {
__global__ void Validate(const double *u, int n, int *bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(u[i]))
    atomicExch(bad, 1);
}

__global__ void DiffuseBoundary(Field f, double *next, double dt, double viscosity, double length, double speed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.nf)
    return;
  auto face = f.faces[i];
  if (face.aperture == 0) {
    next[i] = 0;
    return;
  }
  double h = sqrt(face.area), lap = 0;
  for (int direction = 0; direction < 3; ++direction)
    for (int sign = -1; sign <= 1; sign += 2) {
      auto p = face.centre;
      p[direction] += sign * h;
      for (int a = 0; a < 3; ++a)
        p[a] = Clamp(p[a], f.origin_[a] + (a == face.axis ? 0 : .5 * h),
                     f.origin_[a] + f.roots[a] * f.root_width_ - (a == face.axis ? 0 : .5 * h));
      double fraction = 0;
      if (direction != face.axis)
        for (int half = -1; half <= 1; half += 2) {
          auto wall = face.centre;
          wall[direction] += sign * .5 * h;
          wall[face.axis] += half * .5 * h;
          int other = f.NativeFace(wall, direction);
          if (other >= 0 && f.faces[other].aperture == 0)
            fraction += .5;
        }
      lap += (1 - fraction) * (f.VirtualFace(p, face.axis) - f.velocity_[i]);
    }
  double value = next[i] + dt * viscosity * lap / (h * h);
  if (length > 0) {
    double column = face.centre[0] - (face.axis == 0 ? 0 : .5 * h);
    double distance = f.origin_[0] + f.roots[0] * f.root_width_ - column;
    if (distance < length && column > f.origin_[0]) {
      double blend = Clamp(1 - distance / length, 0, 1);
      blend *= .25 * blend;
      value += blend * ((face.axis == 0 ? speed : 0) - value);
    }
  }
  if ((face.low < 0 || face.high < 0) && !(face.axis == 0 && face.high < 0))
    value = face.axis == 0 ? speed : 0;
  next[i] = value;
}

__global__ void ScalePressure(double *p, int n, double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] *= scale;
}

// Per-block deterministic sums avoid load atomics and transfer only 12 scalars.
__global__ void LoadPartials(Field f, const double *pressure, double density, double viscosity, double *partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double value[12]{};
  if (i < f.nc && f.cells[i].active) {
    Array<double, 3> normal{}, u{}, area{};
    for (int k = f.offsets[i]; k < f.offsets[i + 1]; ++k) {
      int id = f.incidence[k], j = abs(id) - 1;
      auto face = f.faces[j];
      int a = face.axis;
      double wetted = (id > 0 ? 1 : -1) * face.area * (1 - face.aperture), force = wetted * pressure[i];
      value[a] += force;
      value[3 + (a + 1) % 3] += face.centre[(a + 2) % 3] * force;
      value[3 + (a + 2) % 3] -= face.centre[(a + 1) % 3] * force;
      normal[a] += wetted;
      u[a] += face.area * f.velocity_[j];
      area[a] += face.area;
    }
    double wetted = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (wetted > 0) {
      double dot = 0, speed2 = 0;
      for (int a = 0; a < 3; ++a) {
        normal[a] /= wetted;
        u[a] /= area[a];
        dot += normal[a] * u[a];
      }
      for (int a = 0; a < 3; ++a) {
        u[a] -= dot * normal[a];
        speed2 += u[a] * u[a];
      }
      double wall = WallEffectiveViscosity(float(viscosity), float(sqrt(speed2)), float(.5 * f.cells[i].width));
      for (int a = 0; a < 3; ++a) {
        double force = u[a] * 2 * density * wall * wetted / f.cells[i].width;
        value[6 + a] += force;
        value[9 + (a + 1) % 3] += f.cells[i].centre[(a + 2) % 3] * force;
        value[9 + (a + 2) % 3] -= f.cells[i].centre[(a + 1) % 3] * force;
      }
    }
  }
  __shared__ double sum[256];
  for (int component = 0; component < 12; ++component) {
    sum[threadIdx.x] = value[component];
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
      if (threadIdx.x < stride)
        sum[threadIdx.x] += sum[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      partial[component * gridDim.x + blockIdx.x] = sum[0];
    __syncthreads();
  }
}

__global__ void ReduceLoads(const double *partial, int count, double *loads) {
  __shared__ double sum[256];
  double value = 0;
  for (int i = threadIdx.x; i < count; i += 256)
    value += partial[blockIdx.x * count + i];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    loads[blockIdx.x] = sum[0];
}

__global__ void SpeedPartials(Field f, const double *velocity, double *partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double speed = 0;
  if (i < f.nc && f.cells[i].active) {
    Array<double, 3> u{}, area{};
    for (int k = f.offsets[i]; k < f.offsets[i + 1]; ++k) {
      int id = abs(f.incidence[k]) - 1;
      auto g = f.faces[id];
      u[g.axis] += g.area * velocity[id];
      area[g.axis] += g.area;
    }
    for (int a = 0; a < 3; ++a)
      u[a] /= area[a];
    speed = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (!isfinite(speed))
      speed = INFINITY;
  }
  __shared__ double values[256];
  values[threadIdx.x] = speed;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = Max(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x)
    partial[blockIdx.x] = values[0];
}

__global__ void SpeedMaximum(const double *partial, int n, double *output) {
  double value = 0;
  for (int i = threadIdx.x; i < n; i += 256)
    value = Max(value, partial[i]);
  __shared__ double values[256];
  values[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = Max(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x)
    *output = values[0];
}
}  // namespace

struct GpuTransport::Impl {
  int device{};
  Field fields[2]{};
  int *neighbors{}, *layers{}, *bad{};
  double *result{}, *pressure{}, *partial{}, *loads{};
  double min_area{INFINITY};
  std::vector<void *> allocations;
  std::string error;

  ~Impl() {
    cudaSetDevice(device);
    for (void *p : allocations)
      cudaFree(p);
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  template <class T>
  bool Allocate(T *&p, size_t n) {
    if (!Check(cudaMalloc(&p, n * sizeof(T))))
      return false;
    allocations.push_back(p);
    return true;
  }

  template <class T>
  bool Upload(const T *&p, const std::vector<T> &v) {
    T *q{};
    if (!Allocate(q, v.size()) || !Check(cudaMemcpy(q, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)))
      return false;
    p = q;
    return true;
  }

  bool Valid(const double *u, int count = 0) {
    if (!count)
      count = fields[0].nf;
    int host = 0;
    if (!Check(cudaMemset(bad, 0, sizeof(int))))
      return false;
    Validate<<<(count + 255) / 256, 256>>>(u, count, bad);
    if (!Check(cudaGetLastError()) || !Check(cudaMemcpy(&host, bad, sizeof(int), cudaMemcpyDeviceToHost)))
      return false;
    if (host) {
      error = "Non-finite transport field";
      return false;
    }
    return true;
  }

  bool Prepare(Field f) {
    int fb = (f.nf + 255) / 256, cb = (f.nc + 255) / 256;
    ClearSolid<<<fb, 256>>>(f);
    Extend<<<fb, 256>>>(f, neighbors, layers, 1);
    Extend<<<fb, 256>>>(f, neighbors, layers, 2);
    Centres<<<cb, 256>>>(f);
    Slopes<<<cb, 256>>>(f);
    if (f.halo)
      FillHalos<<<(f.halo_count + 255) / 256, 256>>>(f);
    return Check(cudaGetLastError());
  }
};

GpuTransport::GpuTransport() = default;
GpuTransport::~GpuTransport() = default;

std::string GpuTransport::Error() const {
  return !build_error_.empty() ? build_error_ : (impl_ ? impl_->error : "GPU transport is not built");
}

bool GpuTransport::Build(int device,
                         const Mesh &mesh,
                         const Topology &tree,
                         double width,
                         std::array<double, 3> origin,
                         bool cache_halos) {
  auto next = std::make_unique<Impl>();
  next->device = device;
  auto fail = [&]() {
    build_error_ = next->error.empty() ? "Invalid transport geometry" : next->error;
    return false;
  };
  if (mesh.Cells().empty() || mesh.Cells().size() > size_t(INT_MAX) || mesh.Faces().size() > size_t(INT_MAX / 6) ||
      !(width > 0) || !std::isfinite(width) || tree.Leaves().size() > size_t(INT_MAX))
    return fail();
  for (double p : origin)
    if (!std::isfinite(p))
      return fail();
  auto &f = next->fields[0];
  f.n = mesh.CellsPerBlock();
  f.nc = int(mesh.Cells().size());
  f.nf = int(mesh.Faces().size());
  f.nb = int(tree.Leaves().size());
  f.max_level = tree.MaxLevel();
  f.root_width_ = width;
  if (size_t(f.n) * f.n * f.n * size_t(f.nb) != mesh.Cells().size())
    return fail();
  for (int a = 0; a < 3; ++a) {
    f.origin_[a] = origin[a];
    f.roots[a] = tree.Roots()[a];
  }
  std::vector<Cell> cells;
  std::vector<Face> faces;
  for (auto c : mesh.Cells())
    cells.push_back({{c.centre[0], c.centre[1], c.centre[2]}, c.width, c.active});
  for (auto g : mesh.Faces())
    next->min_area = std::min(next->min_area, g.area);
  for (auto g : mesh.Faces())
    faces.push_back({g.low, g.high, g.axis, {g.centre[0], g.centre[1], g.centre[2]}, g.area, g.aperture});
  if (!next->Check(cudaSetDevice(device)) || !next->Upload(f.cells, cells) || !next->Upload(f.faces, faces) ||
      !next->Upload(f.leaves, tree.Leaves()) || !next->Upload(f.offsets, mesh.Offsets()) ||
      !next->Upload(f.incidence, mesh.Incidence()))
    return fail();
  if (cache_halos) {
    size_t side = size_t(f.n) + 5, count = size_t(f.nb) * 3 * side * side * side;
    if (count > size_t(INT_MAX - 255)) {
      next->error = "Transport halo index capacity exceeded";
      return fail();
    }
    f.halo_count = int(count);
    if (!next->Allocate(f.sites, count))
      return fail();
    MapHalos<<<(f.halo_count + 255) / 256, 256>>>(f);
    if (!next->Check(cudaGetLastError()))
      return fail();
  }
  next->fields[1] = f;
  for (auto &field : next->fields)
    if (!next->Allocate(field.velocity_.local, f.nf) || !next->Allocate(field.centres_.local, f.nc) ||
        !next->Allocate(field.slopes_.local, f.nc) || !next->Allocate(field.limits_.local, f.nc) ||
        (cache_halos && !next->Allocate(field.halo, f.halo_count)))
      return fail();
  if (!next->Allocate(next->neighbors, size_t(f.nf) * 6) || !next->Allocate(next->layers, f.nf) ||
      !next->Allocate(next->bad, 1) || !next->Allocate(next->result, f.nf) || !next->Allocate(next->pressure, f.nc) ||
      !next->Allocate(next->partial, size_t((f.nc + 255) / 256) * 12) || !next->Allocate(next->loads, 12))
    return fail();
  SlipNeighbors<<<(f.nf + 255) / 256, 256>>>(f, next->neighbors, next->layers);
  if (!next->Check(cudaGetLastError()) || !next->Check(cudaDeviceSynchronize()))
    return fail();
  impl_ = std::move(next);
  build_error_.clear();
  return true;
}

bool GpuTransport::AdvectDevice(const double *velocity,
                                double *destination,
                                size_t faces,
                                double dt,
                                bool cubic,
                                bool corrected,
                                std::optional<double> inlet_speed) {
  if (!impl_ || !velocity || !destination || faces != size_t(impl_->fields[0].nf) || !std::isfinite(dt))
    return false;
  if (inlet_speed && !std::isfinite(*inlet_speed))
    return false;
  auto &w = *impl_;
  w.error.clear();
  auto a = w.fields[0], b = w.fields[1];
  a.prescribed_inlet = b.prescribed_inlet = inlet_speed.has_value();
  a.inlet_speed = b.inlet_speed = inlet_speed.value_or(0);
  int blocks = (a.nf + 255) / 256;
  if (!w.Check(cudaSetDevice(w.device)) ||
      !w.Check(cudaMemcpy(a.velocity_.local, velocity, faces * sizeof(double), cudaMemcpyDeviceToDevice)) ||
      !w.Valid(a.velocity_.local) || !w.Prepare(a))
    return false;
  Forward<<<blocks, 256>>>(a, dt, cubic, b.velocity_.local);
  if (!w.Check(cudaGetLastError()) || !w.Valid(b.velocity_.local))
    return false;
  const double *result = b.velocity_.local;
  if (corrected) {
    if (!w.Prepare(b))
      return false;
    Correct<<<blocks, 256>>>(a, b, dt, cubic, w.result);
    result = w.result;
  }
  return w.Check(cudaGetLastError()) && w.Valid(result) &&
         (destination == result ||
          w.Check(cudaMemcpy(destination, result, faces * sizeof(double), cudaMemcpyDeviceToDevice)));
}

AdvanceResult GpuTransport::AdvanceDevice(GpuProjection &projection,
                                          double *velocity,
                                          size_t faces,
                                          double *pressure,
                                          size_t cells,
                                          double dt,
                                          double speed,
                                          bool cubic,
                                          double density,
                                          double wall_viscosity,
                                          double viscosity,
                                          double sponge_length,
                                          double pressure_tolerance) {
  AdvanceResult result;
  if (!impl_ || !velocity || !pressure || velocity == pressure || faces != size_t(impl_->fields[0].nf) ||
      cells != size_t(impl_->fields[0].nc) || !(dt > 0) || !std::isfinite(dt) || !std::isfinite(speed) ||
      !(density > 0) || !std::isfinite(density) || !(wall_viscosity > 0) || !std::isfinite(wall_viscosity) ||
      viscosity < 0 || !std::isfinite(viscosity) || sponge_length < 0 || !std::isfinite(sponge_length) ||
      !(pressure_tolerance > 0) || !std::isfinite(pressure_tolerance) || dt * viscosity / impl_->min_area > 1.0 / 6)
    return result;
  auto &w = *impl_;
  auto transport_start = std::chrono::steady_clock::now();
  if (!AdvectDevice(velocity, w.result, faces, dt, cubic, true, speed))
    return result;
  auto f = w.fields[0];
  int fb = (f.nf + 255) / 256, cb = (f.nc + 255) / 256;
  DiffuseBoundary<<<fb, 256>>>(f, w.result, dt, viscosity, sponge_length, speed);
  if (!w.Check(cudaGetLastError()) || !w.Valid(w.result))
    return result;
  auto pressure_start = std::chrono::steady_clock::now();
  result.transport_seconds = std::chrono::duration<double>(pressure_start - transport_start).count();
  result.projection = projection.ProjectDevice(w.result, faces, w.pressure, cells, pressure_tolerance);
  if (!result.projection.converged) {
    w.error = projection.Error();
    return result;
  }
  ScalePressure<<<cb, 256>>>(w.pressure, f.nc, density / dt);
  if (!w.Check(cudaGetLastError()) || !w.Valid(w.pressure, f.nc))
    return result;
  auto loads_start = std::chrono::steady_clock::now();
  result.pressure_seconds = std::chrono::duration<double>(loads_start - pressure_start).count();
  LoadPartials<<<cb, 256>>>(f, w.pressure, density, wall_viscosity, w.partial);
  ReduceLoads<<<12, 256>>>(w.partial, cb, w.loads);
  double loads[12];
  if (!w.Check(cudaGetLastError()) || !w.Check(cudaMemcpy(loads, w.loads, sizeof(loads), cudaMemcpyDeviceToHost)))
    return result;
  for (double v : loads)
    if (!std::isfinite(v)) {
      w.error = "Non-finite GPU loads";
      return result;
    }
  for (int k = 0; k < 6; ++k) {
    result.pressure_loads[k] = loads[k];
    result.friction_loads[k] = loads[6 + k];
    result.total_loads[k] = loads[k] + loads[6 + k];
  }
  if (!w.Check(cudaMemcpy(velocity, w.result, faces * sizeof(double), cudaMemcpyDeviceToDevice)) ||
      !w.Check(cudaMemcpy(pressure, w.pressure, cells * sizeof(double), cudaMemcpyDeviceToDevice)))
    return result;
  result.loads_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loads_start).count();
  result.advanced = true;
  return result;
}

bool GpuTransport::MaximumSpeedDevice(const double *velocity, size_t faces, double &speed) {
  if (!impl_ || !velocity || faces != size_t(impl_->fields[0].nf))
    return false;
  auto &w = *impl_;
  if (!w.Check(cudaSetDevice(w.device)))
    return false;
  auto f = w.fields[0];
  int blocks = (f.nc + 255) / 256;
  SpeedPartials<<<blocks, 256>>>(f, velocity, w.partial);
  SpeedMaximum<<<1, 256>>>(w.partial, blocks, w.loads);
  double value;
  if (!w.Check(cudaGetLastError()) || !w.Check(cudaMemcpy(&value, w.loads, sizeof(double), cudaMemcpyDeviceToHost)))
    return false;
  if (!std::isfinite(value)) {
    w.error = "Non-finite cell speed";
    return false;
  }
  speed = value;
  return true;
}
}  // namespace wind3d::amr
