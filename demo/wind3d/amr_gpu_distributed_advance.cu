#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "amr_gpu_distributed_advance.h"
#include "amr_gpu_distributed_projection.h"
#include "amr_gpu_distributed_transport.h"

namespace wind3d::amr {
namespace {
struct DeviceScope {
  int previous{-1};

  DeviceScope() {
    cudaGetDevice(&previous);
  }

  ~DeviceScope() {
    if (previous >= 0)
      cudaSetDevice(previous);
  }
};

__global__ void PhysicalPressure(double *pressure, int n, double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    pressure[i] *= scale;
}
}  // namespace

struct GpuDistributedAdvance::Impl {
  GpuDistributedProjection projection;
  GpuDistributedTransport transport;
  std::vector<int> devices;
  std::vector<double *> next_velocity, next_pressure;
  std::vector<cudaStream_t> streams;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < streams.size(); ++r) {
      cudaSetDevice(devices[r]);
      if (streams[r]) {
        cudaStreamSynchronize(streams[r]);
        cudaStreamDestroy(streams[r]);
      }
      cudaFree(next_velocity[r]);
      cudaFree(next_pressure[r]);
    }
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  bool Sync() {
    std::string first;
    for (size_t r = 0; r < streams.size(); ++r)
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaStreamSynchronize(streams[r])))
        if (first.empty())
          first = error;
    if (first.empty())
      return true;
    error = first;
    return false;
  }
};

GpuDistributedAdvance::GpuDistributedAdvance() = default;
GpuDistributedAdvance::~GpuDistributedAdvance() = default;

bool GpuDistributedAdvance::Build(const Mesh &mesh,
                                  const Topology &tree,
                                  const std::vector<int> &devices,
                                  double width,
                                  std::array<double, 3> origin) {
  DeviceScope restore;
  error_.clear();
  auto next = std::make_unique<Impl>();
  if (!next->transport.Build(mesh, tree, devices, width, origin)) {
    error_ = next->transport.Error();
    return false;
  }
  if (!next->projection.Build(mesh, tree, devices)) {
    error_ = next->projection.Error();
    return false;
  }
  auto &layouts = next->projection.Layouts();
  for (size_t r = 0; r < layouts.size(); ++r) {
    auto &layout = layouts[r];
    if (layout.owned_faces != next->transport.OwnedFaces()[r].size() ||
        layout.owned_cells != next->transport.OwnedCells()[r].size() ||
        !std::equal(next->transport.OwnedFaces()[r].begin(), next->transport.OwnedFaces()[r].end(),
                    layout.faces.begin()) ||
        !std::equal(next->transport.OwnedCells()[r].begin(), next->transport.OwnedCells()[r].end(),
                    layout.cells.begin())) {
      error_ = "Transport and projection ownership differ";
      return false;
    }
  }
  next->devices = devices;
  next->streams.resize(devices.size());
  next->next_velocity.resize(devices.size());
  next->next_pressure.resize(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &layout = layouts[r];
    if (!next->Check(cudaSetDevice(devices[r])) ||
        !next->Check(cudaStreamCreateWithFlags(&next->streams[r], cudaStreamNonBlocking)) ||
        !next->Check(cudaMalloc(&next->next_velocity[r], layout.faces.size() * sizeof(double))) ||
        !next->Check(cudaMalloc(&next->next_pressure[r], layout.cells.size() * sizeof(double)))) {
      error_ = next->error;
      return false;
    }
  }
  impl_ = std::move(next);
  return true;
}

const std::vector<PressureLocalLayout> &GpuDistributedAdvance::Layouts() const {
  static const std::vector<PressureLocalLayout> empty;
  return impl_ ? impl_->projection.Layouts() : empty;
}

AdvanceResult GpuDistributedAdvance::Advance(const std::vector<double *> &velocity,
                                             const std::vector<double *> &pressure,
                                             double dt,
                                             double speed,
                                             bool cubic,
                                             double density,
                                             double wall_viscosity,
                                             double viscosity,
                                             double sponge_length,
                                             double pressure_tolerance,
                                             int max_iterations) {
  DeviceScope restore;
  error_.clear();
  AdvanceResult result;
  if (!impl_ || velocity.size() != impl_->devices.size() || pressure.size() != velocity.size() || !(dt > 0) ||
      !std::isfinite(dt) || !std::isfinite(speed) || !(density > 0) || !std::isfinite(density) ||
      !std::isfinite(density / dt) || !(wall_viscosity > 0) || !std::isfinite(wall_viscosity) ||
      !(pressure_tolerance > 0) || !std::isfinite(pressure_tolerance) || max_iterations <= 0 ||
      std::any_of(velocity.begin(), velocity.end(), [](auto p) { return !p; }) ||
      std::any_of(pressure.begin(), pressure.end(),
                  [&](auto p) { return !p || std::find(velocity.begin(), velocity.end(), p) != velocity.end(); })) {
    error_ = "Invalid distributed timestep fields or parameters";
    return result;
  }
  auto &w = *impl_;
  w.error.clear();
  auto fail = [&] {
    auto first = w.error;
    w.Sync();
    error_ = first;
    return result;
  };
  std::vector<const double *> original(velocity.begin(), velocity.end());
  auto transport_start = std::chrono::steady_clock::now();
  if (!w.transport.AdvectDiffuse(original, w.next_velocity, dt, speed, cubic, viscosity, sponge_length)) {
    w.error = w.transport.Error();
    return fail();
  }
  auto pressure_start = std::chrono::steady_clock::now();
  result.transport_seconds = std::chrono::duration<double>(pressure_start - transport_start).count();
  result.projection = w.projection.Project(w.next_velocity, w.next_pressure, pressure_tolerance, max_iterations);
  if (!result.projection.converged) {
    w.error = w.projection.Error().empty() ? "Distributed pressure did not converge" : w.projection.Error();
    return fail();
  }
  const auto &layouts = w.projection.Layouts();
  for (size_t r = 0; r < w.devices.size(); ++r) {
    if (!w.Check(cudaSetDevice(w.devices[r])))
      return fail();
    int n = int(layouts[r].owned_cells);
    PhysicalPressure<<<(n + 255) / 256, 256, 0, w.streams[r]>>>(w.next_pressure[r], n, density / dt);
    if (!w.Check(cudaGetLastError()))
      return fail();
  }
  if (!w.Sync())
    return fail();
  auto loads_start = std::chrono::steady_clock::now();
  result.pressure_seconds = std::chrono::duration<double>(loads_start - pressure_start).count();
  TransportLoads loads;
  std::vector<const double *> physical(w.next_pressure.begin(), w.next_pressure.end());
  if (!w.transport.EvaluateLoads(original, physical, loads, density, wall_viscosity)) {
    w.error = w.transport.Error();
    return fail();
  }
  for (size_t r = 0; r < w.devices.size(); ++r) {
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(velocity[r], w.next_velocity[r], layouts[r].owned_faces * sizeof(double),
                                 cudaMemcpyDeviceToDevice, w.streams[r])) ||
        !w.Check(cudaMemcpyAsync(pressure[r], w.next_pressure[r], layouts[r].owned_cells * sizeof(double),
                                 cudaMemcpyDeviceToDevice, w.streams[r])))
      return fail();
  }
  if (!w.Sync())
    return fail();
  result.pressure_loads = loads.pressure;
  result.friction_loads = loads.friction;
  result.total_loads = loads.total;
  result.loads_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loads_start).count();
  result.advanced = true;
  return result;
}
}  // namespace wind3d::amr
