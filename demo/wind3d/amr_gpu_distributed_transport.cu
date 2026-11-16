#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <set>

#include "amr_gpu_distributed_transport.h"
#include "amr_gpu_transport_kernels.cuh"
#include "amr_partition.h"

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

__global__ void ValidateOwned(const double *values, int n, int *bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(values[i]))
    atomicExch(bad, 1);
}
}  // namespace

struct GpuDistributedTransport::Impl {
  struct Rank {
    Field fields[2]{};
    int *neighbors{}, *layers{}, *bad{};
    double *result{}, *pressure{}, *partial{}, *loads{};
    cudaStream_t stream{};
  };

  std::vector<int> devices;
  std::vector<Rank> ranks;
  std::vector<std::vector<int>> owned_faces, owned_cells;
  enum class Values { Original, Forward, Result, Pressure };
  std::vector<std::pair<int, void *>> allocations;
  std::string error;
  double min_area{INFINITY};

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < ranks.size(); ++r) {
      cudaSetDevice(devices[r]);
      if (ranks[r].stream) {
        cudaStreamSynchronize(ranks[r].stream);
        cudaStreamDestroy(ranks[r].stream);
      }
    }
    for (auto [device, pointer] : allocations) {
      cudaSetDevice(device);
      cudaFree(pointer);
    }
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  template <class T>
  bool Allocate(int device, T *&p, size_t n) {
    if (!Check(cudaSetDevice(device)) || !Check(cudaMalloc(&p, std::max(size_t(1), n) * sizeof(T))))
      return false;
    allocations.push_back({device, p});
    return true;
  }

  template <class T>
  bool Upload(int device, const T *&p, const std::vector<T> &v) {
    T *q{};
    if (!Allocate(device, q, v.size()) || !Check(cudaMemcpy(q, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)))
      return false;
    p = q;
    return true;
  }

  bool Sync() {
    std::string first;
    for (size_t r = 0; r < ranks.size(); ++r)
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaStreamSynchronize(ranks[r].stream)))
        if (first.empty())
          first = error;
    if (first.empty())
      return true;
    error = first;
    return false;
  }

  bool Valid(Values field) {
    for (size_t r = 0; r < ranks.size(); ++r) {
      auto &rank = ranks[r];
      int n = field == Values::Pressure ? rank.fields[0].cell_count : rank.fields[0].face_count;
      const double *values = field == Values::Pressure ? rank.pressure
                             : field == Values::Result ? rank.result
                                                       : rank.fields[field == Values::Original ? 0 : 1].velocity_.local;
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaMemsetAsync(rank.bad, 0, sizeof(int), rank.stream)))
        return false;
      ValidateOwned<<<(n + 255) / 256, 256, 0, rank.stream>>>(values, n, rank.bad);
      if (!Check(cudaGetLastError()))
        return false;
    }
    if (!Sync())
      return false;
    for (size_t r = 0; r < ranks.size(); ++r) {
      int bad = 0;
      if (!Check(cudaSetDevice(devices[r])) ||
          !Check(cudaMemcpy(&bad, ranks[r].bad, sizeof(int), cudaMemcpyDeviceToHost)))
        return false;
      if (bad) {
        error = "Non-finite distributed transport field";
        return false;
      }
    }
    return true;
  }

  bool Prepare(int field, bool reconstruct = true) {
    // Every stage completes on all owners before any peer reads its results.
    for (int stage = 0; stage < (reconstruct ? 5 : 3); ++stage) {
      for (size_t r = 0; r < ranks.size(); ++r) {
        auto &rank = ranks[r];
        auto f = rank.fields[field];
        if (!Check(cudaSetDevice(devices[r])))
          return false;
        int fb = (f.face_count + 255) / 256, cb = (f.cell_count + 255) / 256;
        if (stage == 0)
          ClearSolid<<<fb, 256, 0, rank.stream>>>(f);
        else if (stage < 3)
          Extend<<<fb, 256, 0, rank.stream>>>(f, rank.neighbors, rank.layers, stage);
        else if (stage == 3)
          Centres<<<cb, 256, 0, rank.stream>>>(f);
        else
          Slopes<<<cb, 256, 0, rank.stream>>>(f);
        if (!Check(cudaGetLastError()))
          return false;
      }
      if (!Sync())
        return false;
    }
    return true;
  }
};

GpuDistributedTransport::GpuDistributedTransport() = default;
GpuDistributedTransport::~GpuDistributedTransport() = default;

bool GpuDistributedTransport::Build(const Mesh &mesh,
                                    const Topology &tree,
                                    const std::vector<int> &devices,
                                    double width,
                                    std::array<double, 3> origin) {
  DeviceScope restore;
  error_.clear();
  if (devices.empty() || std::set<int>(devices.begin(), devices.end()).size() != devices.size() || !(width > 0) ||
      !std::isfinite(width) || mesh.Cells().empty() || mesh.Faces().size() > size_t(INT_MAX / 6) ||
      std::any_of(origin.begin(), origin.end(), [](double v) { return !std::isfinite(v); })) {
    error_ = "Invalid distributed transport configuration";
    return false;
  }
  FaceMass mass;
  PressurePartition partition;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size()))) {
    error_ = "Invalid transport partition";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->devices = devices;
  auto fail = [&] {
    error_ = next->error;
    return false;
  };
  int device_count = 0;
  if (!next->Check(cudaGetDeviceCount(&device_count)))
    return fail();
  for (int device : devices)
    if (device < 0 || device >= device_count) {
      error_ = "Invalid transport device";
      return false;
    }
  for (int source : devices)
    for (int destination : devices)
      if (source != destination) {
        int peer = 0;
        if (!next->Check(cudaDeviceCanAccessPeer(&peer, source, destination)))
          return fail();
        if (!peer) {
          error_ = "Transport requires peer-accessible GPUs";
          return false;
        }
        if (!next->Check(cudaSetDevice(source)))
          return fail();
        auto status = cudaDeviceEnablePeerAccess(destination, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
          cudaGetLastError();
        else if (!next->Check(status))
          return fail();
      }
  next->ranks.resize(devices.size());
  next->owned_faces = partition.OwnedFaces();
  next->owned_cells = partition.OwnedCells();
  std::vector<int> face_indices(mesh.Faces().size()), cell_indices(mesh.Cells().size());
  for (size_t r = 0; r < devices.size(); ++r) {
    for (size_t i = 0; i < partition.OwnedFaces()[r].size(); ++i)
      face_indices[partition.OwnedFaces()[r][i]] = int(i);
    for (size_t i = 0; i < partition.OwnedCells()[r].size(); ++i)
      cell_indices[partition.OwnedCells()[r][i]] = int(i);
  }
  std::vector<Cell> cells;
  std::vector<Face> faces;
  for (auto c : mesh.Cells())
    cells.push_back({{c.centre[0], c.centre[1], c.centre[2]}, c.width, c.active});
  for (auto f : mesh.Faces())
    next->min_area = std::min(next->min_area, f.area);
  for (auto f : mesh.Faces())
    faces.push_back({f.low, f.high, f.axis, {f.centre[0], f.centre[1], f.centre[2]}, f.area, f.aperture});
  for (size_t r = 0; r < devices.size(); ++r) {
    int device = devices[r];
    auto &rank = next->ranks[r];
    auto &f = rank.fields[0];
    f.n = mesh.CellsPerBlock();
    f.nc = int(cells.size());
    f.nf = int(faces.size());
    f.nb = int(tree.Leaves().size());
    f.max_level = tree.MaxLevel();
    f.root_width_ = width;
    f.face_count = int(partition.OwnedFaces()[r].size());
    f.cell_count = int(partition.OwnedCells()[r].size());
    for (int a = 0; a < 3; ++a) {
      f.origin_[a] = origin[a];
      f.roots[a] = tree.Roots()[a];
    }
    if (!next->Check(cudaSetDevice(device)) ||
        !next->Check(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking)) ||
        !next->Upload(device, f.cells, cells) || !next->Upload(device, f.faces, faces) ||
        !next->Upload(device, f.leaves, tree.Leaves()) || !next->Upload(device, f.offsets, mesh.Offsets()) ||
        !next->Upload(device, f.incidence, mesh.Incidence()) ||
        !next->Upload(device, f.owned_faces, partition.OwnedFaces()[r]) ||
        !next->Upload(device, f.owned_cells, partition.OwnedCells()[r]) ||
        !next->Upload(device, f.velocity_.owners, partition.FaceOwners()) ||
        !next->Upload(device, f.velocity_.indices, face_indices) ||
        !next->Upload(device, f.centres_.owners, partition.CellOwners()) ||
        !next->Upload(device, f.centres_.indices, cell_indices))
      return fail();
    f.slopes_.owners = f.limits_.owners = f.centres_.owners;
    f.slopes_.indices = f.limits_.indices = f.centres_.indices;
    rank.fields[1] = f;
    for (auto &field : rank.fields)
      if (!next->Allocate(device, field.velocity_.local, f.face_count) ||
          !next->Allocate(device, field.centres_.local, f.cell_count) ||
          !next->Allocate(device, field.slopes_.local, f.cell_count) ||
          !next->Allocate(device, field.limits_.local, f.cell_count))
        return fail();
    if (!next->Allocate(device, rank.neighbors, size_t(f.nf) * 6) || !next->Allocate(device, rank.layers, f.nf) ||
        !next->Allocate(device, rank.bad, 1) || !next->Allocate(device, rank.result, f.face_count) ||
        !next->Allocate(device, rank.pressure, f.cell_count) ||
        !next->Allocate(device, rank.partial, size_t((f.cell_count + 255) / 256) * 12) ||
        !next->Allocate(device, rank.loads, 12))
      return fail();
    // Complete pageable metadata uploads before the nonblocking stream reads.
    if (!next->Check(cudaDeviceSynchronize()))
      return fail();
    SlipNeighbors<<<(f.nf + 255) / 256, 256, 0, rank.stream>>>(f, rank.neighbors, rank.layers);
    if (!next->Check(cudaGetLastError()))
      return fail();
  }
  for (int field = 0; field < 2; ++field) {
    std::vector<double *> velocities;
    std::vector<Array<double, 3> *> centres;
    std::vector<Array<double, 9> *> slopes;
    std::vector<Array<double, 6> *> limits;
    for (auto &rank : next->ranks) {
      auto &f = rank.fields[field];
      velocities.push_back(f.velocity_.local);
      centres.push_back(f.centres_.local);
      slopes.push_back(f.slopes_.local);
      limits.push_back(f.limits_.local);
    }
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &f = next->ranks[r].fields[field];
      int device = devices[r];
      if (!next->Upload(device, f.velocity_.peers, velocities) || !next->Upload(device, f.centres_.peers, centres) ||
          !next->Upload(device, f.slopes_.peers, slopes) || !next->Upload(device, f.limits_.peers, limits) ||
          !next->Check(cudaDeviceSynchronize()))
        return fail();
    }
  }
  if (!next->Sync())
    return fail();
  impl_ = std::move(next);
  return true;
}

const std::vector<std::vector<int>> &GpuDistributedTransport::OwnedFaces() const {
  static const std::vector<std::vector<int>> empty;
  return impl_ ? impl_->owned_faces : empty;
}

const std::vector<std::vector<int>> &GpuDistributedTransport::OwnedCells() const {
  static const std::vector<std::vector<int>> empty;
  return impl_ ? impl_->owned_cells : empty;
}

bool GpuDistributedTransport::EvaluateLoads(const std::vector<const double *> &velocity,
                                            const std::vector<const double *> &pressure,
                                            TransportLoads &loads,
                                            double density,
                                            double wall_viscosity) {
  DeviceScope restore;
  error_.clear();
  if (!impl_ || velocity.size() != impl_->ranks.size() || pressure.size() != velocity.size() || !(density > 0) ||
      !std::isfinite(density) || !(wall_viscosity > 0) || !std::isfinite(wall_viscosity) ||
      std::any_of(velocity.begin(), velocity.end(), [](auto p) { return !p; }) ||
      std::any_of(pressure.begin(), pressure.end(), [](auto p) { return !p; })) {
    error_ = "Invalid distributed load fields or parameters";
    return false;
  }
  auto &w = *impl_;
  w.error.clear();
  auto fail = [&] {
    auto first = w.error;
    w.Sync();
    error_ = first;
    return false;
  };
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(rank.fields[0].velocity_.local, velocity[r], w.owned_faces[r].size() * sizeof(double),
                                 cudaMemcpyDeviceToDevice, rank.stream)) ||
        !w.Check(cudaMemcpyAsync(rank.pressure, pressure[r], w.owned_cells[r].size() * sizeof(double),
                                 cudaMemcpyDeviceToDevice, rank.stream)))
      return fail();
  }
  if (!w.Valid(Impl::Values::Original) || !w.Valid(Impl::Values::Pressure) || !w.Prepare(0, false))
    return fail();
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    auto f = rank.fields[0];
    int blocks = (f.cell_count + 255) / 256;
    if (!w.Check(cudaSetDevice(w.devices[r])))
      return fail();
    LoadPartials<<<blocks, 256, 0, rank.stream>>>(f, rank.pressure, density, wall_viscosity, rank.partial);
    ReduceLoads<<<12, 256, 0, rank.stream>>>(rank.partial, blocks, rank.loads);
    if (!w.Check(cudaGetLastError()))
      return fail();
  }
  if (!w.Sync())
    return fail();
  std::array<double, 12> sum{}, compensation{};
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    double local[12];
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpy(local, w.ranks[r].loads, sizeof(local), cudaMemcpyDeviceToHost)))
      return fail();
    for (int k = 0; k < 12; ++k) {
      if (!std::isfinite(local[k])) {
        w.error = "Non-finite distributed loads";
        return fail();
      }
      double y = local[k] - compensation[k], next = sum[k] + y;
      compensation[k] = (next - sum[k]) - y;
      sum[k] = next;
    }
  }
  TransportLoads next;
  for (int k = 0; k < 6; ++k) {
    next.pressure[k] = sum[k];
    next.friction[k] = sum[6 + k];
    next.total[k] = sum[k] + sum[6 + k];
    if (!std::isfinite(next.pressure[k]) || !std::isfinite(next.friction[k]) || !std::isfinite(next.total[k])) {
      w.error = "Non-finite distributed load reduction";
      return fail();
    }
  }
  loads = next;
  return true;
}

bool GpuDistributedTransport::Advect(const std::vector<const double *> &velocity,
                                     const std::vector<double *> &destination,
                                     double dt,
                                     bool cubic,
                                     bool corrected,
                                     std::optional<double> inlet_speed) {
  return Apply(velocity, destination, dt, cubic, corrected, inlet_speed, std::nullopt);
}

bool GpuDistributedTransport::AdvectDiffuse(const std::vector<const double *> &velocity,
                                            const std::vector<double *> &destination,
                                            double dt,
                                            double speed,
                                            bool cubic,
                                            double viscosity,
                                            double sponge_length) {
  return Apply(velocity, destination, dt, cubic, true, speed, Diffusion{viscosity, sponge_length, speed});
}

bool GpuDistributedTransport::Apply(const std::vector<const double *> &velocity,
                                    const std::vector<double *> &destination,
                                    double dt,
                                    bool cubic,
                                    bool corrected,
                                    std::optional<double> inlet_speed,
                                    std::optional<Diffusion> diffusion) {
  DeviceScope restore;
  error_.clear();
  if (!impl_ || velocity.size() != impl_->ranks.size() || destination.size() != velocity.size() || !std::isfinite(dt) ||
      (inlet_speed && !std::isfinite(*inlet_speed)) ||
      std::any_of(velocity.begin(), velocity.end(), [](auto p) { return !p; }) ||
      std::any_of(destination.begin(), destination.end(), [](auto p) { return !p; })) {
    error_ = "Invalid distributed transport fields or parameters";
    return false;
  }
  if (diffusion && (!corrected || !(dt > 0) || !std::isfinite(diffusion->viscosity) || diffusion->viscosity < 0 ||
                    !std::isfinite(diffusion->sponge_length) || diffusion->sponge_length < 0 ||
                    !std::isfinite(diffusion->speed) || dt * diffusion->viscosity / impl_->min_area > 1.0 / 6)) {
    error_ = "Invalid or unstable distributed diffusion parameters";
    return false;
  }
  auto &w = *impl_;
  w.error.clear();
  auto fail = [&] {
    auto first = w.error;
    w.Sync();
    error_ = first;
    return false;
  };
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    for (auto &f : rank.fields) {
      f.prescribed_inlet = inlet_speed.has_value();
      f.inlet_speed = inlet_speed.value_or(0);
    }
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(rank.fields[0].velocity_.local, velocity[r], w.owned_faces[r].size() * sizeof(double),
                                 cudaMemcpyDeviceToDevice, rank.stream)))
      return fail();
  }
  if (!w.Valid(Impl::Values::Original) || !w.Prepare(0))
    return fail();
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    auto f = rank.fields[0];
    if (!w.Check(cudaSetDevice(w.devices[r])))
      return fail();
    Forward<<<(f.face_count + 255) / 256, 256, 0, rank.stream>>>(f, dt, cubic, rank.fields[1].velocity_.local);
    if (!w.Check(cudaGetLastError()))
      return fail();
  }
  if (!w.Valid(Impl::Values::Forward))
    return fail();
  if (corrected) {
    if (!w.Prepare(1))
      return fail();
    for (size_t r = 0; r < w.ranks.size(); ++r) {
      auto &rank = w.ranks[r];
      auto f = rank.fields[0];
      if (!w.Check(cudaSetDevice(w.devices[r])))
        return fail();
      Correct<<<(f.face_count + 255) / 256, 256, 0, rank.stream>>>(f, rank.fields[1], dt, cubic, rank.result);
      if (!w.Check(cudaGetLastError()))
        return fail();
    }
    if (!w.Valid(Impl::Values::Result))
      return fail();
  }
  if (diffusion) {
    for (size_t r = 0; r < w.ranks.size(); ++r) {
      auto &rank = w.ranks[r];
      auto f = rank.fields[0];
      if (!w.Check(cudaSetDevice(w.devices[r])))
        return fail();
      DiffuseBoundary<<<(f.face_count + 255) / 256, 256, 0, rank.stream>>>(f, rank.result, dt, diffusion->viscosity,
                                                                           diffusion->sponge_length, diffusion->speed);
      if (!w.Check(cudaGetLastError()))
        return fail();
    }
    if (!w.Valid(Impl::Values::Result))
      return fail();
  }
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    const double *result = corrected ? rank.result : rank.fields[1].velocity_.local;
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(destination[r], result, w.owned_faces[r].size() * sizeof(double),
                                 cudaMemcpyDeviceToDevice, rank.stream)))
      return fail();
  }
  if (!w.Sync())
    return fail();
  return true;
}
}  // namespace wind3d::amr
