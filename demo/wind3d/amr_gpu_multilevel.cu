#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "amr_gpu_multilevel.h"

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

struct Matrix {
  int n{};
  int *offsets{}, *columns{};
  double *weights{};
};

__device__ double Product(Matrix a, int i, const double *x) {
  double v = 0;
  for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
    v += a.weights[k] * x[a.columns[k]];
  return v;
}

__global__ void Multiply(Matrix a, const double *x, double *y, bool add) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n)
    y[i] = (add ? y[i] : 0) + Product(a, i, x);
}

__global__ void Smooth(Matrix a, const double *diag, const double *b, const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n)
    y[i] = x[i] + .9 * (b[i] - Product(a, i, x)) / diag[i];
}

__global__ void Residual(Matrix a, const double *b, const double *x, double *r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n)
    r[i] = b[i] - Product(a, i, x);
}

__global__ void Gather(const double *x, const int *indices, int n, double *destination) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    destination[i] = x[indices[i]];
}

using Rows = std::vector<std::vector<std::pair<int, double>>>;
}  // namespace

struct GpuMultilevel::Impl {
  struct Rank {
    int n{};
    size_t size{};
    Matrix a, restriction, prolongation, inverse;
    double *diagonal{}, *rhs{}, *x{}, *scratch{}, *residual{};
  };

  struct Link {
    int source{}, destination{}, count{};
    size_t offset{};
    int *indices{};
  };

  struct Level {
    std::vector<Rank> ranks;
    std::vector<Link> links;
    bool identity{};
  };

  std::vector<int> devices;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> ready;
  cudaGraphExec_t cycle_graph{};
  std::vector<Level> levels;
  std::vector<std::pair<int, void *>> allocations;
  std::string error;

  bool Check(cudaError_t s) {
    if (s == cudaSuccess)
      return true;
    error = cudaGetErrorString(s);
    return false;
  }

  bool Sync() {
    std::string first;
    for (size_t r = 0; r < devices.size(); ++r)
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaStreamSynchronize(streams[r])))
        if (first.empty())
          first = error;
    if (first.empty())
      return true;
    error = first;
    return false;
  }

  // Before peer stores, sources wait for destination consumers to finish.
  // Afterwards, destinations wait for the incoming stores. Record all events
  // first so reusing them never creates a cycle within one dependency phase.
  bool Dependencies(const Level &level, bool before) {
    std::vector<unsigned char> participants(devices.size());
    for (const auto &link : level.links)
      participants[link.source] = participants[link.destination] = 1;
    for (size_t r = 0; r < devices.size(); ++r)
      if (participants[r] && (!Check(cudaSetDevice(devices[r])) || !Check(cudaEventRecord(ready[r], streams[r]))))
        return false;
    for (const auto &link : level.links) {
      int waiter = before ? link.source : link.destination;
      int producer = before ? link.destination : link.source;
      if (!Check(cudaSetDevice(devices[waiter])) || !Check(cudaStreamWaitEvent(streams[waiter], ready[producer], 0)))
        return false;
    }
    return true;
  }

  // Two ranks have no unrelated peers; retain the lower-overhead original path.
  bool Barrier() {
    if (devices.size() > 2)
      return true;
    for (size_t r = 0; r < devices.size(); ++r)
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaEventRecord(ready[r], streams[r])))
        return false;
    for (size_t r = 0; r < devices.size(); ++r) {
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      for (size_t source = 0; source < devices.size(); ++source)
        if (source != r && !Check(cudaStreamWaitEvent(streams[r], ready[source], 0)))
          return false;
    }
    return true;
  }

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < streams.size(); ++r)
      if (streams[r]) {
        cudaSetDevice(devices[r]);
        cudaStreamSynchronize(streams[r]);
        cudaStreamDestroy(streams[r]);
      }
    if (cycle_graph) {
      cudaSetDevice(devices.front());
      cudaGraphExecDestroy(cycle_graph);
    }
    for (size_t r = 0; r < ready.size(); ++r)
      if (ready[r]) {
        cudaSetDevice(devices[r]);
        cudaEventDestroy(ready[r]);
      }
    for (auto [d, p] : allocations) {
      cudaSetDevice(d);
      cudaFree(p);
    }
  }

  template <class T>
  bool Allocate(int d, T *&p, size_t n) {
    if (!n)
      return true;
    if (!Check(cudaSetDevice(d)) || !Check(cudaMalloc(&p, n * sizeof(T))))
      return false;
    allocations.push_back({d, p});
    return true;
  }

  template <class T>
  bool Upload(int d, T *&p, const std::vector<T> &v) {
    return Allocate(d, p, v.size()) &&
           (v.empty() || Check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)));
  }

  bool MatrixBuild(int d, Matrix &a, const Rows &rows) {
    std::vector<int> offsets{0}, columns;
    std::vector<double> weights;
    for (const auto &row : rows) {
      for (auto [j, w] : row) {
        if (j < 0) {
          error = "Missing multilevel import";
          return false;
        }
        columns.push_back(j);
        weights.push_back(w);
      }
      offsets.push_back(int(columns.size()));
    }
    a.n = int(rows.size());
    return Upload(d, a.offsets, offsets) && Upload(d, a.columns, columns) && Upload(d, a.weights, weights);
  }

  bool Exchange(size_t depth, double *Rank::*field) {
    auto &level = levels[depth];
    if (devices.size() > 2 && !Dependencies(level, true))
      return false;
    for (auto &link : level.links) {
      int s = link.source, t = link.destination;
      if (!Check(cudaSetDevice(devices[s])))
        return false;
      // A peer store kernel is capturable across devices; cudaMemcpyPeerAsync
      // is not supported by stream capture on the tested CUDA runtime.
      Gather<<<(link.count + 255) / 256, 256, 0, streams[s]>>>(level.ranks[s].*field, link.indices, link.count,
                                                               (level.ranks[t].*field) + link.offset);
      if (!Check(cudaGetLastError()))
        return false;
    }
    return devices.size() > 2 ? Dependencies(level, false) : Barrier();
  }

  bool Capture() {
    if (!Check(cudaSetDevice(devices.front())) ||
        !Check(cudaStreamBeginCapture(streams.front(), cudaStreamCaptureModeThreadLocal)))
      return false;
    bool ok = Check(cudaEventRecord(ready.front(), streams.front()));
    // Fork the capture into every device stream before any level work begins.
    for (size_t r = 1; ok && r < devices.size(); ++r)
      ok = Check(cudaSetDevice(devices[r])) && Check(cudaStreamWaitEvent(streams[r], ready.front(), 0));
    if (ok)
      ok = Cycle(0);
    // Join all branches, including ranks empty on the coarsest levels.
    for (size_t r = 1; ok && r < devices.size(); ++r)
      ok = Check(cudaSetDevice(devices[r])) && Check(cudaEventRecord(ready[r], streams[r]));
    if (ok)
      ok = Check(cudaSetDevice(devices.front()));
    for (size_t r = 1; ok && r < devices.size(); ++r)
      ok = Check(cudaStreamWaitEvent(streams.front(), ready[r], 0));
    std::string first = ok ? std::string{} : error;
    cudaGraph_t graph{};
    cudaSetDevice(devices.front());
    auto status = cudaStreamEndCapture(streams.front(), &graph);
    if (!Check(status) || !ok) {
      if (!first.empty())
        error = first;
      if (graph)
        cudaGraphDestroy(graph);
      return false;
    }
    ok = Check(cudaGraphInstantiate(&cycle_graph, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);
    return ok;
  }

  bool Cycle(size_t depth) {
    auto &level = levels[depth];
    if (level.identity) {
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &v = level.ranks[r];
        if (!v.n)
          continue;
        if (!Check(cudaSetDevice(devices[r])) ||
            !Check(cudaMemcpyAsync(v.x, v.rhs, v.n * sizeof(double), cudaMemcpyDeviceToDevice, streams[r])))
          return false;
      }
      return Barrier();
    }
    if (depth + 1 == levels.size()) {
      if (!Exchange(depth, &Rank::rhs))
        return false;
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &v = level.ranks[r];
        if (!v.n)
          continue;
        if (!Check(cudaSetDevice(devices[r])))
          return false;
        Multiply<<<(v.n + 255) / 256, 256, 0, streams[r]>>>(v.inverse, v.rhs, v.x, false);
        if (!Check(cudaGetLastError()))
          return false;
      }
      return Barrier();
    }
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = level.ranks[r];
      if (!v.size)
        continue;
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaMemsetAsync(v.x, 0, v.size * sizeof(double), streams[r])))
        return false;
    }
    if (!Barrier())
      return false;
    auto smooth = [&]() {
      if (!Exchange(depth, &Rank::x))
        return false;
      for (size_t r = 0; r < devices.size(); ++r) {
        auto &v = level.ranks[r];
        if (!v.n)
          continue;
        if (!Check(cudaSetDevice(devices[r])))
          return false;
        Smooth<<<(v.n + 255) / 256, 256, 0, streams[r]>>>(v.a, v.diagonal, v.rhs, v.x, v.scratch);
        if (!Check(cudaGetLastError()))
          return false;
      }
      if (!Barrier())
        return false;
      for (auto &v : level.ranks)
        std::swap(v.x, v.scratch);
      return true;
    };
    for (int i = 0; i < 2; ++i)
      if (!smooth())
        return false;
    if (!Exchange(depth, &Rank::x))
      return false;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = level.ranks[r];
      if (!v.n)
        continue;
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      Residual<<<(v.n + 255) / 256, 256, 0, streams[r]>>>(v.a, v.rhs, v.x, v.residual);
      if (!Check(cudaGetLastError()))
        return false;
    }
    if (!Barrier() || !Exchange(depth, &Rank::residual))
      return false;
    auto &coarse = levels[depth + 1];
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = coarse.ranks[r];
      if (!v.n)
        continue;
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      Multiply<<<(v.n + 255) / 256, 256, 0, streams[r]>>>(v.restriction, level.ranks[r].residual, v.rhs, false);
      if (!Check(cudaGetLastError()))
        return false;
    }
    if (!Barrier() || !Cycle(depth + 1) || !Exchange(depth + 1, &Rank::x))
      return false;
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = level.ranks[r];
      if (!v.n)
        continue;
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      Multiply<<<(v.n + 255) / 256, 256, 0, streams[r]>>>(v.prolongation, coarse.ranks[r].x, v.x, true);
      if (!Check(cudaGetLastError()))
        return false;
    }
    if (!Barrier())
      return false;
    for (int i = 0; i < 2; ++i)
      if (!smooth())
        return false;
    return true;
  }
};

GpuMultilevel::GpuMultilevel() = default;
GpuMultilevel::~GpuMultilevel() = default;

bool GpuMultilevel::Build(const PressureHierarchy &hierarchy, const std::vector<int> &devices) {
  error_.clear();
  DeviceScope restore;
  const auto &host = hierarchy.Levels();
  if (host.empty() || devices.empty() || std::set<int>(devices.begin(), devices.end()).size() != devices.size()) {
    error_ = "Invalid multilevel hierarchy/devices";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->devices = devices;
  next->streams.resize(devices.size());
  next->ready.resize(devices.size());
  next->levels.resize(host.size());
  int count = 0;
  if (!next->Check(cudaGetDeviceCount(&count))) {
    error_ = next->error;
    return false;
  }
  for (size_t r = 0; r < devices.size(); ++r)
    if (devices[r] < 0 || devices[r] >= count || !next->Check(cudaSetDevice(devices[r])) ||
        !next->Check(cudaStreamCreateWithFlags(&next->streams[r], cudaStreamNonBlocking)) ||
        !next->Check(cudaEventCreateWithFlags(&next->ready[r], cudaEventDisableTiming))) {
      error_ = "Invalid multilevel device or stream";
      return false;
    }
  // Global-to-local indices are only retained while assembling device CSR.
  std::vector<std::vector<std::vector<int>>> indices(host.size());
  for (size_t d = 0; d < host.size(); ++d) {
    const auto &h = host[d];
    auto &level = next->levels[d];
    if (h.owned_cells.size() != devices.size()) {
      error_ = "Multilevel rank count mismatch";
      return false;
    }
    level.identity = h.identity;
    level.ranks.resize(devices.size());
    indices[d].resize(devices.size(), std::vector<int>(h.owners.size(), -1));
    std::map<std::pair<int, int>, std::set<int>> imports;
    for (const auto &e : h.exchanges)
      imports[{e.source, e.destination}].insert(e.cells.begin(), e.cells.end());
    if (d + 1 == host.size() && !h.identity) {
      if (h.owners.size() > 64) {
        error_ = "Coarse direct solve exceeds 64 cells";
        return false;
      }
      for (size_t r = 0; r < devices.size(); ++r)
        if (!h.owned_cells[r].empty())
          for (size_t i = 0; i < h.owners.size(); ++i)
            if (h.owners[i] != int(r))
              imports[{h.owners[i], int(r)}].insert(int(i));
    }
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = level.ranks[r];
      v.n = int(h.owned_cells[r].size());
      v.size = v.n;
      for (int i = 0; i < v.n; ++i)
        indices[d][r][h.owned_cells[r][i]] = i;
    }
    for (const auto &[pair, ids] : imports) {
      if (ids.empty())
        continue;
      int s = pair.first, t = pair.second, peer = 0;
      if (!next->Check(cudaDeviceCanAccessPeer(&peer, devices[s], devices[t])) || !peer) {
        error_ = "Multilevel requires peer-accessible GPUs";
        return false;
      }
      Impl::Link link;
      link.source = s;
      link.destination = t;
      link.count = int(ids.size());
      link.offset = level.ranks[t].size;
      std::vector<int> gather;
      for (int i : ids) {
        gather.push_back(indices[d][s][i]);
        indices[d][t][i] = int(level.ranks[t].size++);
      }
      if (!next->Check(cudaSetDevice(devices[s]))) {
        error_ = next->error;
        return false;
      }
      auto access = cudaDeviceEnablePeerAccess(devices[t], 0);
      if (access == cudaErrorPeerAccessAlreadyEnabled)
        cudaGetLastError();
      else if (!next->Check(access)) {
        error_ = next->error;
        return false;
      }
      if (!next->Upload(devices[s], link.indices, gather)) {
        error_ = next->error;
        return false;
      }
      level.links.push_back(link);
    }
  }
  for (size_t d = 0; d < host.size(); ++d) {
    const auto &h = host[d];
    size_t n = h.owners.size();
    std::vector<double> inverse;
    if (d + 1 == host.size() && !h.identity) {
      std::vector<double> a(n * n);
      inverse.resize(n * n);
      double scale = 0;
      for (size_t i = 0; i < n; ++i) {
        inverse[i * n + i] = 1;
        for (int k = h.offsets[i]; k < h.offsets[i + 1]; ++k) {
          a[i * n + h.columns[k]] = h.weights[k];
          scale = std::max(scale, std::abs(h.weights[k]));
        }
      }
      for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        for (size_t i = k + 1; i < n; ++i)
          if (std::abs(a[i * n + k]) > std::abs(a[pivot * n + k]))
            pivot = i;
        if (std::abs(a[pivot * n + k]) <= 1e-14 * scale) {
          error_ = "Singular multilevel coarse matrix";
          return false;
        }
        for (size_t j = 0; j < n; ++j) {
          std::swap(a[k * n + j], a[pivot * n + j]);
          std::swap(inverse[k * n + j], inverse[pivot * n + j]);
        }
        double v = a[k * n + k];
        for (size_t j = 0; j < n; ++j) {
          a[k * n + j] /= v;
          inverse[k * n + j] /= v;
        }
        for (size_t i = 0; i < n; ++i)
          if (i != k) {
            double f = a[i * n + k];
            for (size_t j = 0; j < n; ++j) {
              a[i * n + j] -= f * a[k * n + j];
              inverse[i * n + j] -= f * inverse[k * n + j];
            }
          }
      }
      if (std::any_of(inverse.begin(), inverse.end(), [](double v) { return !std::isfinite(v); })) {
        error_ = "Nonfinite coarse inverse";
        return false;
      }
    }
    for (size_t r = 0; r < devices.size(); ++r) {
      auto &v = next->levels[d].ranks[r];
      const auto &ids = h.owned_cells[r];
      Rows a(v.n), restriction(v.n), prolongation(v.n), inv(v.n);
      std::vector<double> diag;
      for (int i = 0; i < v.n; ++i) {
        int id = ids[i];
        diag.push_back(h.diagonal[id]);
        for (int k = h.offsets[id]; k < h.offsets[id + 1]; ++k)
          a[i].push_back({indices[d][r][h.columns[k]], h.weights[k]});
        if (!h.parent.empty() && h.parent[id] >= 0)
          prolongation[i].push_back({indices[d + 1][r][h.parent[id]], 1});
        if (!inverse.empty())
          for (size_t j = 0; j < n; ++j)
            inv[i].push_back({indices[d][r][j], inverse[id * n + j]});
      }
      if (d > 0)
        for (size_t i = 0; i < host[d - 1].parent.size(); ++i) {
          int parent = host[d - 1].parent[i];
          if (parent >= 0 && h.owners[parent] == int(r))
            restriction[indices[d][r][parent]].push_back({indices[d - 1][r][i], 1});
        }
      int device = devices[r];
      if (!next->MatrixBuild(device, v.a, a) || !next->MatrixBuild(device, v.restriction, restriction) ||
          !next->MatrixBuild(device, v.prolongation, prolongation) || !next->MatrixBuild(device, v.inverse, inv) ||
          !next->Upload(device, v.diagonal, diag) || !next->Allocate(device, v.rhs, v.size) ||
          !next->Allocate(device, v.x, v.size) || !next->Allocate(device, v.scratch, v.size) ||
          !next->Allocate(device, v.residual, v.size)) {
        error_ = next->error;
        return false;
      }
    }
  }
  // Host uploads may stage asynchronously; finish each device before publishing.
  for (int device : devices)
    if (!next->Check(cudaSetDevice(device)) || !next->Check(cudaDeviceSynchronize())) {
      error_ = next->error;
      return false;
    }
  if (!next->Capture()) {
    error_ = next->error;
    return false;
  }
  impl_ = std::move(next);
  return true;
}

bool GpuMultilevel::Apply(const std::vector<double *> &input, const std::vector<double *> &output) {
  error_.clear();
  if (!impl_ || input.size() != impl_->devices.size() || output.size() != input.size() ||
      std::any_of(input.begin(), input.end(), [](auto p) { return !p; }) ||
      std::any_of(output.begin(), output.end(),
                  [&](auto p) { return !p || std::find(input.begin(), input.end(), p) != input.end(); })) {
    error_ = "Invalid multilevel fields";
    return false;
  }
  DeviceScope restore;
  auto &w = *impl_;
  w.error.clear();
  auto fail = [&]() {
    auto first = w.error;
    w.Sync();
    error_ = first;
    return false;
  };
  for (size_t r = 0; r < input.size(); ++r) {
    auto &v = w.levels.front().ranks[r];
    if (!v.n)
      continue;
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(v.rhs, input[r], v.n * sizeof(double), cudaMemcpyDeviceToDevice, w.streams[r])))
      return fail();
  }
  // Input copies are outside the graph; drain them before the root launch.
  // The captured fork/join dependencies order every internal device branch.
  if (!w.Sync() || !w.Check(cudaSetDevice(w.devices.front())) ||
      !w.Check(cudaGraphLaunch(w.cycle_graph, w.streams.front())) || !w.Check(cudaStreamSynchronize(w.streams.front())))
    return fail();
  for (size_t r = 0; r < output.size(); ++r) {
    auto &v = w.levels.front().ranks[r];
    if (!v.n)
      continue;
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(output[r], v.x, v.n * sizeof(double), cudaMemcpyDeviceToDevice, w.streams[r])))
      return fail();
  }
  if (!w.Sync())
    return fail();
  return true;
}
}  // namespace wind3d::amr
