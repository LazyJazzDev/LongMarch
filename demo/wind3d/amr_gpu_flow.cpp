#include "amr_gpu_flow.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"
#include "amr_transfer.h"

namespace wind3d::amr {
struct GpuFlow::Impl {
  int device{}, n{};
  double width{};
  bool compatible_pressure{};
  std::array<double, 3> origin{};
  Distance distance;
  Topology tree;
  Mesh mesh;
  GpuProjection projection;
  GpuTransport transport;
  double *u{}, *p{};
  FlowInfo info;
  std::string error;

  ~Impl() {
    cudaSetDevice(device);
    cudaFree(u);
    cudaFree(p);
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
    FaceGradient gradient;
    if (!compatible_pressure && !gradient.Build(mesh)) {
      error = "Adaptive gradient construction failed";
      return false;
    }
    if (!projection.Build(device, mesh, gradient, true, true, compatible_pressure)) {
      error = projection.Error();
      return false;
    }
    if (!transport.Build(device, mesh, tree, width, origin)) {
      error = transport.Error();
      return false;
    }
    info.cells = mesh.Cells().size();
    info.faces = mesh.Faces().size();
    info.blocks = tree.Leaves().size();
    return Check(cudaSetDevice(device)) && Check(cudaMalloc(&u, info.faces * sizeof(double))) &&
           Check(cudaMalloc(&p, info.cells * sizeof(double))) && Check(cudaMemset(p, 0, info.cells * sizeof(double)));
  }

  bool DownloadVelocity(std::vector<double> &velocity) {
    velocity.resize(info.faces);
    return Check(cudaSetDevice(device)) &&
           Check(cudaMemcpy(velocity.data(), u, info.faces * sizeof(double), cudaMemcpyDeviceToHost));
  }

  bool UploadVelocity(const std::vector<double> &velocity) {
    return velocity.size() == info.faces && Check(cudaSetDevice(device)) &&
           Check(cudaMemcpy(u, velocity.data(), info.faces * sizeof(double), cudaMemcpyHostToDevice));
  }
};

GpuFlow::GpuFlow() = default;
GpuFlow::~GpuFlow() = default;

std::string GpuFlow::Error() const {
  return error_;
}

FlowInfo GpuFlow::Info() const {
  return impl_ ? impl_->info : FlowInfo{};
}

bool GpuFlow::Initialize(int device,
                         const Topology &tree,
                         int n,
                         double width,
                         std::array<double, 3> origin,
                         const Distance &distance,
                         double speed,
                         bool compatible_pressure) {
  if (!std::isfinite(speed)) {
    error_ = "Invalid freestream speed";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->device = device;
  next->compatible_pressure = compatible_pressure;
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

bool GpuFlow::Regrid(const Topology &tree) {
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
  next->device = impl_->device;
  next->compatible_pressure = impl_->compatible_pressure;
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
  if (!impl_->DownloadVelocity(old_velocity)) {
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
  auto projection = next->projection.ProjectDevice(next->u, next->info.faces, next->p, next->info.cells);
  if (!projection.converged) {
    error_ = "Regrid projection did not converge: " + next->projection.Error();
    return false;
  }
  if (!next->Check(cudaMemset(next->p, 0, next->info.cells * sizeof(double))) ||
      !next->Check(cudaDeviceSynchronize())) {
    error_ = next->error;
    return false;
  }
  next->info.steps = impl_->info.steps;
  next->info.time = impl_->info.time;
  next->info.generation = impl_->info.generation + 1;
  impl_ = std::move(next);
  error_.clear();
  return true;
}

bool GpuFlow::Adapt(const RefinementOptions &options, RefinementReport &report) {
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  std::vector<double> velocity;
  if (!impl_->DownloadVelocity(velocity)) {
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

AdvanceResult GpuFlow::Advance(double dt, double speed, const FlowParameters &parameters) {
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return {};
  }
  if (impl_->info.steps == std::numeric_limits<uint64_t>::max() || !std::isfinite(impl_->info.time + dt)) {
    error_ = "Simulation clock overflow";
    return {};
  }
  auto result =
      impl_->transport.AdvanceDevice(impl_->projection, impl_->u, impl_->info.faces, impl_->p, impl_->info.cells, dt,
                                     speed, parameters.cubic, parameters.density, parameters.wall_viscosity,
                                     parameters.viscosity, parameters.sponge_length, parameters.pressure_tolerance);
  if (!result.advanced) {
    error_ = "GPU step failed: " + impl_->transport.Error();
    return result;
  }
  ++impl_->info.steps;
  impl_->info.time += dt;
  impl_->info.pressure_valid = true;
  error_.clear();
  return result;
}

bool GpuFlow::MaximumSpeed(double &speed) {
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  if (!impl_->transport.MaximumSpeedDevice(impl_->u, impl_->info.faces, speed)) {
    error_ = impl_->transport.Error();
    return false;
  }
  error_.clear();
  return true;
}

bool GpuFlow::Restore(const FlowSnapshot &snapshot) {
  const auto &info = snapshot.info;
  auto finite = [](const auto &values) {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
  };
  if (!impl_ || snapshot.compatible_pressure != impl_->compatible_pressure ||
      snapshot.topology.Roots() != impl_->tree.Roots() || snapshot.mesh.CellsPerBlock() != impl_->n ||
      info.cells != snapshot.mesh.Cells().size() || info.faces != snapshot.mesh.Faces().size() ||
      info.blocks != snapshot.topology.Leaves().size() || snapshot.velocity.size() != info.faces ||
      snapshot.pressure.size() != info.cells || !std::isfinite(info.time) || info.time < 0 ||
      !finite(snapshot.velocity) || !finite(snapshot.pressure)) {
    error_ = "Invalid or incompatible flow snapshot";
    return false;
  }

  struct DeviceRestore {
    int previous{-1};

    ~DeviceRestore() {
      if (previous >= 0)
        cudaSetDevice(previous);
    }
  } restore;

  auto status = cudaGetDevice(&restore.previous);
  if (status != cudaSuccess) {
    error_ = cudaGetErrorString(status);
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->device = impl_->device;
  next->n = impl_->n;
  next->width = impl_->width;
  next->origin = impl_->origin;
  next->distance = impl_->distance;
  next->compatible_pressure = impl_->compatible_pressure;
  next->tree = snapshot.topology;
  if (!next->Build()) {
    error_ = next->error;
    return false;
  }
  auto cell_equal = [](const MeshCell &a, const MeshCell &b) {
    return a.centre == b.centre && a.width == b.width && a.active == b.active;
  };
  auto face_equal = [](const MeshFace &a, const MeshFace &b) {
    return a.low == b.low && a.high == b.high && a.axis == b.axis && a.centre == b.centre && a.area == b.area &&
           a.distance == b.distance && a.aperture == b.aperture;
  };
  if (next->info.cells != info.cells || next->info.faces != info.faces ||
      next->mesh.Offsets() != snapshot.mesh.Offsets() || next->mesh.Incidence() != snapshot.mesh.Incidence() ||
      !std::equal(next->mesh.Cells().begin(), next->mesh.Cells().end(), snapshot.mesh.Cells().begin(), cell_equal) ||
      !std::equal(next->mesh.Faces().begin(), next->mesh.Faces().end(), snapshot.mesh.Faces().begin(), face_equal)) {
    error_ = "Snapshot geometry differs from the initialized domain/body";
    return false;
  }
  if (!next->UploadVelocity(snapshot.velocity) ||
      !next->Check(
          cudaMemcpy(next->p, snapshot.pressure.data(), info.cells * sizeof(double), cudaMemcpyHostToDevice)) ||
      !next->Check(cudaDeviceSynchronize())) {
    error_ = next->error;
    return false;
  }
  next->info = info;
  impl_ = std::move(next);
  error_.clear();
  return true;
}

bool GpuFlow::Readback(FlowSnapshot &snapshot) const {
  if (!impl_) {
    error_ = "GPU flow is not initialized";
    return false;
  }
  FlowSnapshot next;
  next.compatible_pressure = impl_->compatible_pressure;
  next.info = impl_->info;
  next.topology = impl_->tree;
  next.mesh = impl_->mesh;
  next.pressure.resize(next.info.cells);
  if (!impl_->DownloadVelocity(next.velocity) ||
      !impl_->Check(
          cudaMemcpy(next.pressure.data(), impl_->p, next.info.cells * sizeof(double), cudaMemcpyDeviceToHost))) {
    error_ = impl_->error;
    return false;
  }
  snapshot = std::move(next);
  error_.clear();
  return true;
}
}  // namespace wind3d::amr
