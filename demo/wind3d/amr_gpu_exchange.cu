#include <cuda_runtime.h>

#include <algorithm>
#include <set>

#include "amr_gpu_exchange.h"

namespace wind3d::amr {
namespace {
__global__ void Gather(const double *values, const int *indices, int count, double *packed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    packed[i] = values[indices[i]];
}

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

struct GpuPressureExchange::Impl {
  struct Batch {
    int count{};
    size_t destination_offset{};
    int *indices{};
    double *packed{};
  };

  struct Link {
    int source{}, destination{};
    Batch cells, faces;
  };

  std::vector<int> devices;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> ready;
  std::vector<PressureLocalLayout> layouts;
  std::vector<Link> links;
  std::vector<std::pair<int, void *>> allocations;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t i = 0; i < streams.size(); ++i) {
      if (streams[i]) {
        cudaSetDevice(devices[i]);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
      }
    }
    for (size_t r = 0; r < ready.size(); ++r)
      if (ready[r]) {
        cudaSetDevice(devices[r]);
        cudaEventDestroy(ready[r]);
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
  bool Allocate(int device, T *&pointer, size_t count) {
    if (!count)
      return true;
    if (!Check(cudaSetDevice(device)) || !Check(cudaMalloc(&pointer, count * sizeof(T))))
      return false;
    allocations.push_back({device, pointer});
    return true;
  }

  bool BatchBuild(Batch &batch,
                  int source,
                  const std::vector<int> &ids,
                  const std::vector<int> &local_indices,
                  std::vector<int> &destination) {
    batch.count = int(ids.size());
    batch.destination_offset = destination.size();
    destination.insert(destination.end(), ids.begin(), ids.end());
    if (ids.empty())
      return true;
    std::vector<int> indices;
    for (int id : ids)
      indices.push_back(local_indices[id]);
    int device = devices[source];
    return Allocate(device, batch.indices, ids.size()) && Allocate(device, batch.packed, ids.size()) &&
           Check(cudaMemcpy(batch.indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
};

GpuPressureExchange::GpuPressureExchange() = default;
GpuPressureExchange::~GpuPressureExchange() = default;

bool GpuPressureExchange::Build(const PressurePartition &partition, const std::vector<int> &devices) {
  error_.clear();
  DeviceScope restore;
  if (devices.empty() || devices.size() != partition.OwnedCells().size() ||
      std::set<int>(devices.begin(), devices.end()).size() != devices.size()) {
    error_ = "Distinct devices must match the pressure partition ranks";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->devices = devices;
  next->streams.resize(devices.size());
  next->ready.resize(devices.size());
  next->layouts.resize(devices.size());
  int count = 0;
  if (!next->Check(cudaGetDeviceCount(&count))) {
    error_ = next->error;
    return false;
  }
  for (int device : devices)
    if (device < 0 || device >= count) {
      error_ = "Invalid pressure exchange device";
      return false;
    }
  std::vector<int> cell_index(partition.CellOwners().size()), face_index(partition.FaceOwners().size());
  for (size_t r = 0; r < devices.size(); ++r) {
    auto &layout = next->layouts[r];
    layout.cells = partition.OwnedCells()[r];
    layout.faces = partition.OwnedFaces()[r];
    layout.owned_cells = layout.cells.size();
    layout.owned_faces = layout.faces.size();
    for (size_t i = 0; i < layout.cells.size(); ++i)
      cell_index[layout.cells[i]] = int(i);
    for (size_t i = 0; i < layout.faces.size(); ++i)
      face_index[layout.faces[i]] = int(i);
    if (!next->Check(cudaSetDevice(devices[r])) ||
        !next->Check(cudaStreamCreateWithFlags(&next->streams[r], cudaStreamNonBlocking)) ||
        !next->Check(cudaEventCreateWithFlags(&next->ready[r], cudaEventDisableTiming))) {
      error_ = next->error;
      return false;
    }
  }
  for (const auto &exchange : partition.Exchanges()) {
    int source = exchange.source, destination = exchange.destination, peer = 0;
    if (!next->Check(cudaDeviceCanAccessPeer(&peer, devices[source], devices[destination]))) {
      error_ = next->error;
      return false;
    }
    if (!peer) {
      error_ = "Pressure exchange requires peer-accessible GPUs";
      return false;
    }
    // Explicit peer copies do not require changing the process-wide peer-access
    // enable state, which may also be used by other solvers in this process.
    Impl::Link link;
    link.source = source;
    link.destination = destination;
    if (!next->BatchBuild(link.cells, source, exchange.cells, cell_index, next->layouts[destination].cells) ||
        !next->BatchBuild(link.faces, source, exchange.faces, face_index, next->layouts[destination].faces)) {
      error_ = next->error;
      return false;
    }
    next->links.push_back(link);
  }
  impl_ = std::move(next);
  return true;
}

const std::vector<PressureLocalLayout> &GpuPressureExchange::Layouts() const {
  static const std::vector<PressureLocalLayout> empty;
  return impl_ ? impl_->layouts : empty;
}

bool GpuPressureExchange::Exchange(const std::vector<double *> &values, bool faces) {
  error_.clear();
  if (!impl_ || values.size() != impl_->devices.size() ||
      std::any_of(values.begin(), values.end(), [](auto p) { return p == nullptr; })) {
    error_ = "Pressure exchange needs one allocated field per rank";
    return false;
  }
  DeviceScope restore;
  auto &w = *impl_;
  bool ok = true;
  for (const auto &link : w.links) {
    auto &batch = faces ? link.faces : link.cells;
    if (!batch.count)
      continue;
    int source = w.devices[link.source], destination = w.devices[link.destination];
    if (!w.Check(cudaSetDevice(source))) {
      ok = false;
      break;
    }
    auto stream = w.streams[link.source];
    Gather<<<(batch.count + 255) / 256, 256, 0, stream>>>(values[link.source], batch.indices, batch.count,
                                                          batch.packed);
    if (!w.Check(cudaGetLastError()) ||
        !w.Check(cudaMemcpyPeerAsync(values[link.destination] + batch.destination_offset, destination, batch.packed,
                                     source, batch.count * sizeof(double), stream))) {
      ok = false;
      break;
    }
  }
  // Drain every stream even after an enqueue failure before buffers can be reused.
  std::string failure = ok ? std::string{} : w.error;
  for (size_t r = 0; r < w.devices.size(); ++r)
    if (!w.Check(cudaSetDevice(w.devices[r])) || !w.Check(cudaStreamSynchronize(w.streams[r]))) {
      if (failure.empty())
        failure = w.error;
      ok = false;
    }
  if (!ok)
    error_ = failure;
  return ok;
}

bool GpuPressureExchange::EnqueueFaces(const std::vector<double *> &values, const std::vector<cudaStream_t> &streams) {
  error_.clear();
  if (!impl_ || values.size() != impl_->devices.size() || streams.size() != values.size() ||
      std::any_of(values.begin(), values.end(), [](auto p) { return !p; })) {
    error_ = "Face enqueue needs one field and stream per rank";
    return false;
  }
  DeviceScope restore;
  auto &w = *impl_;
  auto barrier = [&] {
    if (w.devices.size() == 1)
      return true;
    for (size_t r = 0; r < streams.size(); ++r)
      if (!w.Check(cudaSetDevice(w.devices[r])) || !w.Check(cudaEventRecord(w.ready[r], streams[r])))
        return false;
    for (size_t r = 0; r < streams.size(); ++r) {
      if (!w.Check(cudaSetDevice(w.devices[r])))
        return false;
      for (size_t source = 0; source < streams.size(); ++source)
        if (source != r && !w.Check(cudaStreamWaitEvent(streams[r], w.ready[source], 0)))
          return false;
    }
    return true;
  };
  auto fail = [&] {
    error_ = w.error;
    return false;
  };
  if (!barrier())
    return fail();
  for (const auto &link : w.links) {
    const auto &batch = link.faces;
    if (!batch.count)
      continue;
    int source = w.devices[link.source], destination = w.devices[link.destination];
    if (!w.Check(cudaSetDevice(source)))
      return fail();
    auto stream = streams[link.source];
    Gather<<<(batch.count + 255) / 256, 256, 0, stream>>>(values[link.source], batch.indices, batch.count,
                                                          batch.packed);
    if (!w.Check(cudaGetLastError()) ||
        !w.Check(cudaMemcpyPeerAsync(values[link.destination] + batch.destination_offset, destination, batch.packed,
                                     source, batch.count * sizeof(double), stream)))
      return fail();
  }
  if (!barrier())
    return fail();
  return true;
}

bool GpuPressureExchange::ExchangeCells(const std::vector<double *> &values) {
  return Exchange(values, false);
}

bool GpuPressureExchange::ExchangeFaces(const std::vector<double *> &values) {
  return Exchange(values, true);
}
}  // namespace wind3d::amr
