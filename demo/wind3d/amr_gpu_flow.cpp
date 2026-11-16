#include "amr_gpu_flow.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"
#include "amr_transfer.h"

namespace wind3d::amr {
namespace {
constexpr uint64_t checkpoint_magic = 0x57494e4433414d52ULL;

uint64_t Bits(double value) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double Value(uint64_t bits) {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct CheckpointIO {
  std::istream *input{};
  std::ostream *output{};
  uint64_t hash{14695981039346656037ULL};

  bool Word(uint64_t &value) {
    unsigned char bytes[8]{};
    if (output) {
      for (int i = 0; i < 8; ++i)
        bytes[i] = static_cast<unsigned char>(value >> (8 * i));
      if (!output->write(reinterpret_cast<const char *>(bytes), 8))
        return false;
    } else {
      if (!input->read(reinterpret_cast<char *>(bytes), 8))
        return false;
      value = 0;
      for (int i = 0; i < 8; ++i)
        value |= uint64_t(bytes[i]) << (8 * i);
    }
    for (auto byte : bytes)
      hash = (hash ^ byte) * 1099511628211ULL;
    return true;
  }
};

uint64_t GeometryHash(const Mesh &mesh) {
  uint64_t hash = 14695981039346656037ULL;
  auto word = [&](uint64_t value) {
    for (int i = 0; i < 8; ++i)
      hash = (hash ^ ((value >> (8 * i)) & 255)) * 1099511628211ULL;
  };
  word(mesh.CellsPerBlock());
  for (const auto &cell : mesh.Cells()) {
    for (double value : cell.centre)
      word(Bits(value));
    word(Bits(cell.width));
    word(cell.active);
  }
  for (const auto &face : mesh.Faces()) {
    word(uint64_t(face.low));
    word(uint64_t(face.high));
    word(face.axis);
    for (double value : face.centre)
      word(Bits(value));
    word(Bits(face.area));
    word(Bits(face.distance));
    word(Bits(face.aperture));
  }
  for (int value : mesh.Offsets())
    word(uint64_t(value));
  for (int value : mesh.Incidence())
    word(uint64_t(value));
  return hash;
}
}  // namespace

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

bool GpuFlow::SaveCheckpoint(const std::string &path) const {
  namespace fs = std::filesystem;
  FlowSnapshot snapshot;
  if (path.empty() || !Readback(snapshot)) {
    if (path.empty())
      error_ = "Empty checkpoint path";
    return false;
  }
  const auto &info = snapshot.info;
  auto roots = snapshot.topology.Roots();
  std::array<uint64_t, 20> header{checkpoint_magic,
                                  1,
                                  uint64_t(snapshot.compatible_pressure),
                                  uint64_t(impl_->n),
                                  uint64_t(roots[0]),
                                  uint64_t(roots[1]),
                                  uint64_t(roots[2]),
                                  uint64_t(snapshot.topology.MaxLevel()),
                                  uint64_t(info.blocks),
                                  uint64_t(info.faces),
                                  uint64_t(info.cells),
                                  info.steps,
                                  info.generation,
                                  uint64_t(info.pressure_valid),
                                  GeometryHash(snapshot.mesh),
                                  Bits(impl_->width),
                                  Bits(impl_->origin[0]),
                                  Bits(impl_->origin[1]),
                                  Bits(impl_->origin[2]),
                                  Bits(info.time)};
  static std::atomic<uint64_t> serial{};
  fs::path temporary;
  std::error_code ec;
  bool created = false;
  for (int attempt = 0; attempt < 16 && !created; ++attempt) {
    temporary = path + ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
                std::to_string(serial.fetch_add(1));
    created = fs::create_directory(temporary, ec);
    if (ec)
      break;
  }
  if (!created) {
    error_ = "Cannot create checkpoint temporary directory: " + ec.message();
    return false;
  }
  auto payload = temporary / "state";
  std::ofstream file(payload, std::ios::binary | std::ios::trunc);
  CheckpointIO io{nullptr, &file};
  bool ok = bool(file);
  for (auto &value : header)
    if (ok)
      ok = io.Word(value);
  for (auto block : snapshot.topology.Leaves())
    for (int coordinate : {block.level, block.x, block.y, block.z}) {
      uint64_t value = uint64_t(coordinate);
      if (ok)
        ok = io.Word(value);
    }
  for (const auto *values : {&snapshot.velocity, &snapshot.pressure})
    for (double scalar : *values) {
      uint64_t value = Bits(scalar);
      if (ok)
        ok = io.Word(value);
    }
  uint64_t checksum = io.hash;
  if (ok)
    ok = io.Word(checksum);
  file.flush();
  ok = ok && bool(file);
  file.close();
  ok = ok && !file.fail();
  if (ok) {
    fs::rename(payload, fs::path(path), ec);
    ok = !ec;
  }
  std::error_code cleanup;
  fs::remove(payload, cleanup);
  fs::remove(temporary, cleanup);
  if (!ok) {
    error_ = "Checkpoint write/replacement failed";
    return false;
  }
  error_.clear();
  return true;
}

bool GpuFlow::LoadCheckpoint(const std::string &path) {
  namespace fs = std::filesystem;
  auto invalid = [&]() {
    error_ = "Invalid, truncated or incompatible checkpoint";
    return false;
  };
  if (!impl_)
    return invalid();
  std::error_code ec;
  uintmax_t bytes = fs::file_size(fs::path(path), ec);
  if (ec || bytes < 168)
    return invalid();
  std::ifstream file(path, std::ios::binary);
  CheckpointIO io{&file, nullptr};
  std::array<uint64_t, 20> h{};
  for (auto &value : h)
    if (!io.Word(value))
      return invalid();
  auto roots = impl_->tree.Roots();
  if (h[0] != checkpoint_magic || h[1] != 1 || h[2] != uint64_t(impl_->compatible_pressure) ||
      h[3] != uint64_t(impl_->n) || h[4] != uint64_t(roots[0]) || h[5] != uint64_t(roots[1]) ||
      h[6] != uint64_t(roots[2]) || h[7] > 12 || h[13] > 1 || h[15] != Bits(impl_->width) ||
      h[16] != Bits(impl_->origin[0]) || h[17] != Bits(impl_->origin[1]) || h[18] != Bits(impl_->origin[2]))
    return invalid();
  // Validate exact payload size without overflowing arithmetic or trusting
  // encoded allocation sizes. The runtime uses signed 32-bit mesh indexing.
  uintmax_t remaining = bytes - 168;
  if (!h[8] || h[8] > remaining / 32 || h[8] > uint64_t(INT_MAX))
    return invalid();
  remaining -= h[8] * 32;
  if (!h[9] || h[9] > remaining / 8 || h[9] > uint64_t(INT_MAX / 6))
    return invalid();
  remaining -= h[9] * 8;
  if (!h[10] || h[10] > remaining / 8 || h[10] > uint64_t(INT_MAX) || remaining != h[10] * 8)
    return invalid();
  std::vector<Block> blocks(static_cast<size_t>(h[8]));
  for (auto &block : blocks) {
    int *coordinates[]{&block.level, &block.x, &block.y, &block.z};
    for (auto coordinate : coordinates) {
      uint64_t value = 0;
      if (!io.Word(value) || value > uint64_t(INT_MAX))
        return invalid();
      *coordinate = int(value);
    }
  }
  FlowSnapshot snapshot;
  snapshot.compatible_pressure = impl_->compatible_pressure;
  snapshot.info = {size_t(h[10]), size_t(h[9]), size_t(h[8]), h[11], h[12], Value(h[19]), bool(h[13])};
  if (!snapshot.topology.Restore(roots, int(h[7]), blocks.size(), blocks) ||
      !snapshot.mesh.Build(snapshot.topology, impl_->n, impl_->width, impl_->origin) ||
      (impl_->distance && !snapshot.mesh.Embed(impl_->distance)) || snapshot.mesh.Cells().size() != h[10] ||
      snapshot.mesh.Faces().size() != h[9] || GeometryHash(snapshot.mesh) != h[14])
    return invalid();
  snapshot.velocity.resize(size_t(h[9]));
  snapshot.pressure.resize(size_t(h[10]));
  for (auto *values : {&snapshot.velocity, &snapshot.pressure})
    for (double &value : *values) {
      uint64_t bits = 0;
      if (!io.Word(bits))
        return invalid();
      value = Value(bits);
    }
  uint64_t expected = io.hash, checksum = 0;
  if (!io.Word(checksum) || checksum != expected || file.peek() != std::char_traits<char>::eof())
    return invalid();
  return Restore(snapshot);
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
