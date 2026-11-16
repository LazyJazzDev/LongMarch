#include "amr_gpu_distributed_flow.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#include "amr_gpu_distributed_advance.h"
#include "amr_transfer.h"

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
}  // namespace

struct GpuDistributedFlow::Impl {
  std::vector<int> devices;
  int n{};
  double width{};
  std::array<double, 3> origin{};
  Distance distance;
  Topology tree;
  Mesh mesh;
  GpuDistributedAdvance stepper;
  std::vector<double *> u, p;
  FlowInfo info;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < u.size(); ++r) {
      cudaSetDevice(devices[r]);
      cudaFree(u[r]);
      cudaFree(p[r]);
    }
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  bool Build(const Mesh *previous = nullptr) {
    if (!mesh.Build(tree, n, width, origin)) {
      error = "Invalid adaptive mesh";
      return false;
    }
    if (distance && !mesh.Embed(distance, previous)) {
      error = "Body intersects a refinement interface";
      return false;
    }
    if (!stepper.Build(mesh, tree, devices, width, origin)) {
      error = stepper.Error();
      return false;
    }
    info.cells = mesh.Cells().size();
    info.faces = mesh.Faces().size();
    info.blocks = tree.Leaves().size();
    u.resize(devices.size());
    p.resize(devices.size());
    const auto &layouts = stepper.Layouts();
    for (size_t r = 0; r < devices.size(); ++r) {
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaMalloc(&u[r], layouts[r].faces.size() * sizeof(double))) ||
          !Check(cudaMalloc(&p[r], layouts[r].cells.size() * sizeof(double))) ||
          !Check(cudaMemset(p[r], 0, layouts[r].cells.size() * sizeof(double))) || !Check(cudaDeviceSynchronize()))
        return false;
    }
    return true;
  }

  bool Download(std::vector<double> &values, bool cells) {
    values.resize(cells ? info.cells : info.faces);
    const auto &layouts = stepper.Layouts();
    for (size_t r = 0; r < devices.size(); ++r) {
      const auto &layout = layouts[r];
      size_t count = cells ? layout.owned_cells : layout.owned_faces;
      const auto &ids = cells ? layout.cells : layout.faces;
      std::vector<double> local(count);
      if (!Check(cudaSetDevice(devices[r])) ||
          !Check(cudaMemcpy(local.data(), cells ? p[r] : u[r], count * sizeof(double), cudaMemcpyDeviceToHost)))
        return false;
      for (size_t i = 0; i < count; ++i)
        values[ids[i]] = local[i];
    }
    return true;
  }

  bool UploadVelocity(const std::vector<double> &velocity) {
    if (velocity.size() != info.faces) {
      error = "Invalid remapped velocity size";
      return false;
    }
    const auto &layouts = stepper.Layouts();
    for (size_t r = 0; r < devices.size(); ++r) {
      const auto &layout = layouts[r];
      std::vector<double> local(layout.owned_faces);
      for (size_t i = 0; i < local.size(); ++i)
        local[i] = velocity[layout.faces[i]];
      if (!Check(cudaSetDevice(devices[r])) ||
          !Check(cudaMemcpy(u[r], local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice)) ||
          !Check(cudaDeviceSynchronize()))
        return false;
    }
    return true;
  }
};

GpuDistributedFlow::GpuDistributedFlow() = default;
GpuDistributedFlow::~GpuDistributedFlow() = default;

std::string GpuDistributedFlow::Error() const {
  return error_;
}

FlowInfo GpuDistributedFlow::Info() const {
  return impl_ ? impl_->info : FlowInfo{};
}

bool GpuDistributedFlow::Initialize(const std::vector<int> &devices,
                                    const Topology &tree,
                                    int n,
                                    double width,
                                    std::array<double, 3> origin,
                                    const Distance &distance,
                                    double speed) {
  DeviceScope restore;
  if (!std::isfinite(speed)) {
    error_ = "Invalid freestream speed";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->devices = devices;
  next->tree = tree;
  next->n = n;
  next->width = width;
  next->origin = origin;
  next->distance = distance;
  if (!next->Build()) {
    error_ = next->error;
    return false;
  }
  std::vector<double> velocity(next->info.faces);
  for (size_t f = 0; f < velocity.size(); ++f)
    velocity[f] = next->mesh.Faces()[f].axis == 0 && next->mesh.Faces()[f].aperture > 0 ? speed : 0;
  if (!next->UploadVelocity(velocity)) {
    error_ = next->error;
    return false;
  }
  impl_ = std::move(next);
  error_.clear();
  return true;
}

bool GpuDistributedFlow::Regrid(const Topology &tree) {
  DeviceScope restore;
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  if (tree.Roots() != impl_->tree.Roots()) {
    error_ = "Regridding cannot change the physical domain";
    return false;
  }
  if (tree.MaxLevel() == impl_->tree.MaxLevel() && tree.Leaves() == impl_->tree.Leaves()) {
    error_.clear();
    return true;
  }
  if (impl_->info.generation == std::numeric_limits<uint64_t>::max()) {
    error_ = "Grid generation overflow";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->devices = impl_->devices;
  next->n = impl_->n;
  next->width = impl_->width;
  next->origin = impl_->origin;
  next->distance = impl_->distance;
  next->tree = tree;
  if (!next->Build(&impl_->mesh)) {
    error_ = next->error;
    return false;
  }
  std::vector<double> old_velocity, mapped;
  if (!impl_->Download(old_velocity, false)) {
    error_ = impl_->error;
    return false;
  }
  if (!TransferVelocity(impl_->mesh, impl_->tree, old_velocity, next->mesh, tree, next->width, next->origin, mapped)) {
    error_ = "Conservative transfer failed; body resolution/open areas must be preserved";
    return false;
  }
  if (!next->UploadVelocity(mapped)) {
    error_ = next->error;
    return false;
  }
  auto projection = next->stepper.ProjectVelocity(next->u);
  if (!projection.converged) {
    error_ = "Regrid projection did not converge: " + next->stepper.Error();
    return false;
  }
  next->info.steps = impl_->info.steps;
  next->info.time = impl_->info.time;
  next->info.generation = impl_->info.generation + 1;
  impl_ = std::move(next);
  error_.clear();
  return true;
}

bool GpuDistributedFlow::Adapt(const RefinementOptions &options, RefinementReport &report) {
  DeviceScope restore;
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  std::vector<double> velocity;
  if (!impl_->Download(velocity, false)) {
    error_ = impl_->error;
    return false;
  }
  Topology next;
  RefinementReport pending;
  if (!PlanRefinement(impl_->mesh, impl_->tree, velocity, impl_->width, impl_->origin, options, next, pending)) {
    error_ = "Refinement planning failed: invalid data or capacity exhausted";
    return false;
  }
  if (!Regrid(next))
    return false;
  report = pending;
  return true;
}

AdvanceResult GpuDistributedFlow::Advance(double dt, double speed, const FlowParameters &parameters) {
  DeviceScope restore;
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return {};
  }
  if (impl_->info.steps == std::numeric_limits<uint64_t>::max() || !std::isfinite(impl_->info.time + dt)) {
    error_ = "Simulation clock overflow";
    return {};
  }
  auto result = impl_->stepper.Advance(impl_->u, impl_->p, dt, speed, parameters.cubic, parameters.density,
                                       parameters.wall_viscosity, parameters.viscosity, parameters.sponge_length,
                                       parameters.pressure_tolerance);
  if (!result.advanced) {
    error_ = "GPU step failed: " + impl_->stepper.Error();
    return result;
  }
  ++impl_->info.steps;
  impl_->info.time += dt;
  impl_->info.pressure_valid = true;
  error_.clear();
  return result;
}

bool GpuDistributedFlow::MaximumSpeed(double &speed) {
  DeviceScope restore;
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  if (!impl_->stepper.MaximumSpeed(impl_->u, speed)) {
    error_ = impl_->stepper.Error();
    return false;
  }
  error_.clear();
  return true;
}

bool GpuDistributedFlow::Readback(FlowSnapshot &snapshot) const {
  DeviceScope restore;
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  FlowSnapshot next;
  next.info = impl_->info;
  next.topology = impl_->tree;
  next.mesh = impl_->mesh;
  if (!impl_->Download(next.velocity, false) || !impl_->Download(next.pressure, true)) {
    error_ = impl_->error;
    return false;
  }
  snapshot = std::move(next);
  error_.clear();
  return true;
}
}  // namespace wind3d::amr
