#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>

#include "amr_gpu_operators.h"

namespace wind3d::amr {
namespace {
struct Matrix {
  int rows{};
  int *offsets{}, *columns{};
  double *weights{};
};

__global__ void Multiply(Matrix a, const double *input, double *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.rows)
    return;
  double value = 0;
  for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
    value += a.weights[k] * input[a.columns[k]];
  output[i] = value;
}

constexpr double mass_centre = 1.25, mass_radius = .76;
constexpr int mass_iterations = 32;

__global__ void InitializeInverse(int n,
                                  const unsigned char *coupled,
                                  const double *scale,
                                  const double *rhs,
                                  double *b,
                                  double *x,
                                  double *scratch,
                                  double *direction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    b[i] = coupled[i] ? rhs[i] * scale[i] : 0;
    x[i] = direction[i] = b[i] / mass_centre;
    // Queued exchanges include uncoupled faces too; keep both buffers defined.
    scratch[i] = 0;
  }
}

__global__ void IterateInverse(Matrix a,
                               const int *rows,
                               const double *b,
                               const double *x,
                               double *next,
                               double *direction,
                               double alpha,
                               double beta) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows)
    return;
  int i = rows[row];
  double product = 0;
  for (int k = a.offsets[row]; k < a.offsets[row + 1]; ++k)
    product += a.weights[k] * x[a.columns[k]];
  double d = alpha * (b[i] - product) + beta * direction[i];
  next[i] = x[i] + d;
  direction[i] = d;
}

__global__ void FinishInverse(int n,
                              const unsigned char *free,
                              const unsigned char *coupled,
                              const double *scale,
                              const double *diagonal,
                              const double *rhs,
                              const double *x,
                              double *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = !free[i] ? 0 : coupled[i] ? scale[i] * x[i] : rhs[i] / diagonal[i];
}

// Each local row reads its owning RHS once. The 31-ring neighborhood contains
// the complete dependency cone of the fixed degree-31 inverse polynomial.
__global__ void InitializeNeighborhood(int n,
                                       const int *owners,
                                       const int *indices,
                                       const double *scale,
                                       double *const *rhs,
                                       double *b,
                                       double *x,
                                       double *direction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    b[i] = rhs[owners[i]][indices[i]] * scale[i];
    x[i] = direction[i] = b[i] / mass_centre;
  }
}

__global__ void FinishNeighborhood(int n,
                                   const unsigned char *free,
                                   const int *rows,
                                   const double *scale,
                                   const double *diagonal,
                                   const double *rhs,
                                   const double *x,
                                   double *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = !free[i] ? 0 : rows[i] >= 0 ? scale[i] * x[rows[i]] : rhs[i] / diagonal[i];
}

__global__ void PinInactive(int n, const unsigned char *active, const double *input, double *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !active[i])
    output[i] = input[i];
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

struct GpuPressureOperators::Impl {
  struct Rank {
    Matrix d, transpose, mass, normalized, two_point, neighborhood;
    int *neighbor_owners{}, *neighbor_indices{}, *neighbor_rows{}, *owned_neighbors{};
    double *neighbor_scale{}, *neighbor_b{}, *neighbor_x{}, *neighbor_scratch{}, *neighbor_direction{};
    double **peer_rhs{};
    cudaGraphExec_t local_inverse{};
    int *coupled_rows{};
    unsigned char *free{}, *coupled{}, *active{};
    double *gradient{}, *flux{}, *inverse_rhs{}, *inverse_result{};
    cudaEvent_t captured{};
    double *scale{}, *diagonal{}, *b{}, *x{}, *scratch{}, *direction{};
    cudaStream_t stream{};
  };

  GpuPressureExchange exchange;
  std::vector<int> devices;
  std::vector<Rank> ranks;
  bool coupled{};
  cudaGraphExec_t inverse_graph{};
  std::vector<std::pair<int, void *>> allocations;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < ranks.size(); ++r)
      if (ranks[r].stream) {
        cudaSetDevice(devices[r]);
        cudaStreamSynchronize(ranks[r].stream);
        cudaStreamDestroy(ranks[r].stream);
        if (ranks[r].local_inverse)
          cudaGraphExecDestroy(ranks[r].local_inverse);
      }
    if (inverse_graph) {
      cudaSetDevice(devices.front());
      cudaGraphExecDestroy(inverse_graph);
    }
    for (size_t r = 0; r < ranks.size(); ++r)
      if (ranks[r].captured) {
        cudaSetDevice(devices[r]);
        cudaEventDestroy(ranks[r].captured);
      }
    for (auto [device, p] : allocations) {
      cudaSetDevice(device);
      cudaFree(p);
    }
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  template <class T>
  bool Upload(int device, T *&pointer, const std::vector<T> &values) {
    if (values.empty())
      return true;
    if (!Check(cudaSetDevice(device)) || !Check(cudaMalloc(&pointer, values.size() * sizeof(T))))
      return false;
    allocations.push_back({device, pointer});
    return Check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <class T>
  bool Allocate(int device, T *&pointer, size_t count) {
    if (!Check(cudaSetDevice(device)) || !Check(cudaMalloc(&pointer, count * sizeof(T))))
      return false;
    allocations.push_back({device, pointer});
    return true;
  }

  bool Synchronize() {
    std::string failure;
    for (size_t r = 0; r < ranks.size(); ++r)
      if (!Check(cudaSetDevice(devices[r])) || !Check(cudaStreamSynchronize(ranks[r].stream)))
        if (failure.empty())
          failure = error;
    if (failure.empty())
      return true;
    error = failure;
    return false;
  }

  bool BuildNeighborhoods(const FaceMass &mass,
                          const PressurePartition &partition,
                          const std::vector<unsigned char> &coupled_mask,
                          const std::vector<double> &global_scale) {
    for (int source : devices)
      for (int destination : devices)
        if (source != destination) {
          int accessible = 0;
          if (!Check(cudaDeviceCanAccessPeer(&accessible, source, destination)))
            return false;
          if (!accessible) {
            error = "Mass neighborhoods require mutually peer-accessible GPUs";
            return false;
          }
          if (!Check(cudaSetDevice(source)))
            return false;
          auto status = cudaDeviceEnablePeerAccess(destination, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
          else if (!Check(status))
            return false;
        }
    std::vector<int> owner_indices(coupled_mask.size());
    std::vector<double *> pointers;
    for (size_t r = 0; r < ranks.size(); ++r) {
      pointers.push_back(ranks[r].inverse_rhs);
      const auto &ids = partition.OwnedFaces()[r];
      for (size_t i = 0; i < ids.size(); ++i)
        owner_indices[ids[i]] = int(i);
    }
    for (size_t r = 0; r < ranks.size(); ++r) {
      auto &rank = ranks[r];
      int device = devices[r];
      std::vector<int> depth(coupled_mask.size(), -1), ids, local(coupled_mask.size(), -1);
      for (int id : partition.OwnedFaces()[r])
        if (coupled_mask[id]) {
          depth[id] = 0;
          ids.push_back(id);
        }
      for (size_t head = 0; head < ids.size(); ++head) {
        int id = ids[head];
        if (depth[id] == mass_iterations - 1)
          continue;
        for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k) {
          int j = mass.Columns()[k];
          if (coupled_mask[j] && depth[j] < 0) {
            depth[j] = depth[id] + 1;
            ids.push_back(j);
          }
        }
      }
      for (size_t i = 0; i < ids.size(); ++i)
        local[ids[i]] = int(i);
      std::vector<int> owners, indices, rows, owned;
      std::vector<double> scales;
      std::vector<std::vector<std::pair<int, double>>> matrix(ids.size());
      for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        owners.push_back(partition.FaceOwners()[id]);
        indices.push_back(owner_indices[id]);
        rows.push_back(int(i));
        scales.push_back(global_scale[id]);
        for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k) {
          int j = mass.Columns()[k];
          if (local[j] >= 0)
            matrix[i].push_back({local[j], mass.Weights()[k] * global_scale[id] * global_scale[j]});
        }
      }
      for (int id : partition.OwnedFaces()[r])
        owned.push_back(local[id]);
      size_t count = std::max(size_t(1), ids.size());
      if (!MatrixBuild(device, rank.neighborhood, matrix) || !Upload(device, rank.neighbor_owners, owners) ||
          !Upload(device, rank.neighbor_indices, indices) || !Upload(device, rank.neighbor_rows, rows) ||
          !Upload(device, rank.owned_neighbors, owned) || !Upload(device, rank.neighbor_scale, scales) ||
          !Upload(device, rank.peer_rhs, pointers) || !Allocate(device, rank.neighbor_b, count) ||
          !Allocate(device, rank.neighbor_x, count) || !Allocate(device, rank.neighbor_scratch, count) ||
          !Allocate(device, rank.neighbor_direction, count) || !Check(cudaDeviceSynchronize()))
        return false;
      if (!Check(cudaStreamBeginCapture(rank.stream, cudaStreamCaptureModeThreadLocal)))
        return false;
      int n = rank.neighborhood.rows;
      bool ok = true;
      double *current = rank.neighbor_x, *next = rank.neighbor_scratch;
      if (n) {
        InitializeNeighborhood<<<(n + 255) / 256, 256, 0, rank.stream>>>(
            n, rank.neighbor_owners, rank.neighbor_indices, rank.neighbor_scale, rank.peer_rhs, rank.neighbor_b,
            current, rank.neighbor_direction);
        ok = Check(cudaGetLastError());
        double rho = mass_radius / mass_centre;
        for (int step = 1; ok && step < mass_iterations; ++step) {
          double updated = 1 / (2 * mass_centre / mass_radius - rho);
          IterateInverse<<<(n + 255) / 256, 256, 0, rank.stream>>>(
              rank.neighborhood, rank.neighbor_rows, rank.neighbor_b, current, next, rank.neighbor_direction,
              2 * updated / mass_radius, updated * rho);
          ok = Check(cudaGetLastError());
          std::swap(current, next);
          rho = updated;
        }
      }
      if (ok) {
        FinishNeighborhood<<<(rank.mass.rows + 255) / 256, 256, 0, rank.stream>>>(
            rank.mass.rows, rank.free, rank.owned_neighbors, rank.scale, rank.diagonal, rank.inverse_rhs, current,
            rank.inverse_result);
        ok = Check(cudaGetLastError());
      }
      std::string first = error;
      cudaGraph_t graph{};
      auto status = cudaStreamEndCapture(rank.stream, &graph);
      if (!Check(status) || !ok) {
        if (!ok)
          error = first;
        if (graph)
          cudaGraphDestroy(graph);
        return false;
      }
      ok = Check(cudaGraphInstantiate(&rank.local_inverse, graph, nullptr, nullptr, 0));
      cudaGraphDestroy(graph);
      if (!ok)
        return false;
    }
    return true;
  }

  bool RunInverse() {
    std::vector<double *> current, next;
    std::vector<cudaStream_t> streams;
    for (size_t r = 0; r < ranks.size(); ++r) {
      auto &rank = ranks[r];
      streams.push_back(rank.stream);
      current.push_back(rank.x);
      next.push_back(rank.scratch);
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      if (coupled) {
        InitializeInverse<<<(rank.mass.rows + 255) / 256, 256, 0, rank.stream>>>(
            rank.mass.rows, rank.coupled, rank.scale, rank.inverse_rhs, rank.b, rank.x, rank.scratch, rank.direction);
        if (!Check(cudaGetLastError()))
          return false;
      }
    }
    if (coupled) {
      double rho = mass_radius / mass_centre;
      for (int step = 1; step < mass_iterations; ++step) {
        if (!exchange.EnqueueFaces(current, streams)) {
          error = exchange.Error();
          return false;
        }
        double updated = 1 / (2 * mass_centre / mass_radius - rho);
        for (size_t r = 0; r < ranks.size(); ++r) {
          auto &rank = ranks[r];
          if (!Check(cudaSetDevice(devices[r])))
            return false;
          if (rank.normalized.rows)
            IterateInverse<<<(rank.normalized.rows + 255) / 256, 256, 0, rank.stream>>>(
                rank.normalized, rank.coupled_rows, rank.b, current[r], next[r], rank.direction,
                2 * updated / mass_radius, updated * rho);
          if (!Check(cudaGetLastError()))
            return false;
        }
        current.swap(next);
        rho = updated;
      }
    }
    for (size_t r = 0; r < ranks.size(); ++r) {
      auto &rank = ranks[r];
      if (!Check(cudaSetDevice(devices[r])))
        return false;
      FinishInverse<<<(rank.mass.rows + 255) / 256, 256, 0, rank.stream>>>(rank.mass.rows, rank.free, rank.coupled,
                                                                           rank.scale, rank.diagonal, rank.inverse_rhs,
                                                                           current[r], rank.inverse_result);
      if (!Check(cudaGetLastError()))
        return false;
    }
    return true;
  }

  bool CaptureInverse() {
    if (!Check(cudaSetDevice(devices.front())) ||
        !Check(cudaStreamBeginCapture(ranks.front().stream, cudaStreamCaptureModeThreadLocal)))
      return false;
    bool ok = Check(cudaEventRecord(ranks.front().captured, ranks.front().stream));
    for (size_t r = 1; ok && r < ranks.size(); ++r)
      ok = Check(cudaSetDevice(devices[r])) && Check(cudaStreamWaitEvent(ranks[r].stream, ranks.front().captured, 0));
    if (ok)
      ok = RunInverse();
    for (size_t r = 1; ok && r < ranks.size(); ++r)
      ok = Check(cudaSetDevice(devices[r])) && Check(cudaEventRecord(ranks[r].captured, ranks[r].stream));
    if (ok)
      ok = Check(cudaSetDevice(devices.front()));
    for (size_t r = 1; ok && r < ranks.size(); ++r)
      ok = Check(cudaStreamWaitEvent(ranks.front().stream, ranks[r].captured, 0));
    std::string first = ok ? std::string{} : error;
    cudaGraph_t graph{};
    cudaSetDevice(devices.front());
    auto status = cudaStreamEndCapture(ranks.front().stream, &graph);
    if (!Check(status) || !ok) {
      if (!first.empty())
        error = first;
      if (graph)
        cudaGraphDestroy(graph);
      return false;
    }
    ok = Check(cudaGraphInstantiate(&inverse_graph, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);
    return ok;
  }

  bool MatrixBuild(int device, Matrix &a, const std::vector<std::vector<std::pair<int, double>>> &rows) {
    std::vector<int> offsets{0}, columns;
    std::vector<double> weights;
    for (const auto &row : rows) {
      for (auto [column, weight] : row) {
        if (column < 0 || columns.size() >= size_t(INT_MAX)) {
          error = "Invalid local pressure matrix";
          return false;
        }
        columns.push_back(column);
        weights.push_back(weight);
      }
      offsets.push_back(int(columns.size()));
    }
    a.rows = int(rows.size());
    return Upload(device, a.offsets, offsets) && Upload(device, a.columns, columns) &&
           Upload(device, a.weights, weights);
  }
};

GpuPressureOperators::GpuPressureOperators() = default;
GpuPressureOperators::~GpuPressureOperators() = default;

bool GpuPressureOperators::Build(const Mesh &mesh,
                                 const Topology &tree,
                                 const std::vector<int> &devices,
                                 bool pressure_boundaries) {
  error_.clear();
  DeviceScope restore;
  FaceMass mass;
  PressurePartition partition;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size()))) {
    error_ = "Invalid mesh or pressure partition";
    return false;
  }
  auto next = std::make_unique<Impl>();
  std::vector<unsigned char> free(mesh.Faces().size()), coupled(mesh.Faces().size());
  std::vector<double> scale(mesh.Faces().size());
  for (size_t i = 0; i < free.size(); ++i) {
    const auto &f = mesh.Faces()[i];
    free[i] = f.aperture > 0 && (!pressure_boundaries || (f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0));
  }
  for (size_t i = 0; i < free.size(); ++i)
    if (free[i])
      for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
        if (mass.Columns()[k] != int(i) && free[mass.Columns()[k]]) {
          coupled[i] = 1;
          next->coupled = true;
          const auto &f = mesh.Faces()[i];
          scale[i] = 1 / std::sqrt(f.area * f.distance * f.aperture);
          break;
        }
  if (!next->exchange.Build(partition, devices, coupled)) {
    error_ = next->exchange.Error();
    return false;
  }
  next->devices = devices;
  next->ranks.resize(devices.size());
  for (size_t r = 0; r < devices.size(); ++r) {
    const auto &layout = next->exchange.Layouts()[r];
    std::vector<int> cells(mesh.Cells().size(), -1), faces(mesh.Faces().size(), -1);
    for (size_t i = 0; i < layout.cells.size(); ++i)
      cells[layout.cells[i]] = int(i);
    for (size_t i = 0; i < layout.faces.size(); ++i)
      faces[layout.faces[i]] = int(i);
    using Rows = std::vector<std::vector<std::pair<int, double>>>;
    Rows two_point(layout.owned_cells);
    Rows d(layout.owned_cells), transpose(layout.owned_faces), m(layout.owned_faces), normalized;
    std::vector<int> coupled_rows;
    std::vector<unsigned char> local_free, local_coupled;
    std::vector<double> local_scale, diagonal;
    std::vector<unsigned char> active;
    for (size_t i = 0; i < layout.owned_cells; ++i) {
      int cell = layout.cells[i];
      active.push_back(mesh.Cells()[cell].active);
      std::map<int, double> pressure_row;
      if (!mesh.Cells()[cell].active)
        pressure_row[cell] = 1;
      for (int k = mesh.Offsets()[cell]; k < mesh.Offsets()[cell + 1]; ++k) {
        int entry = mesh.Incidence()[k], id = std::abs(entry) - 1;
        const auto &f = mesh.Faces()[id];
        d[i].push_back({faces[id], (entry > 0 ? -1 : 1) * f.area * f.aperture});
        if (mesh.Cells()[cell].active && free[id]) {
          double weight = f.area * f.aperture / f.distance;
          pressure_row[cell] += weight;
          int neighbor = f.low == cell ? f.high : f.low;
          if (neighbor >= 0)
            pressure_row[neighbor] -= weight;
        }
      }
      for (auto [neighbor, weight] : pressure_row)
        two_point[i].push_back({cells[neighbor], weight});
    }
    for (size_t i = 0; i < layout.owned_faces; ++i) {
      int id = layout.faces[i];
      const auto &f = mesh.Faces()[id];
      local_free.push_back(free[id]);
      local_coupled.push_back(coupled[id]);
      local_scale.push_back(scale[id]);
      diagonal.push_back(mass.Diagonal()[id]);
      if (coupled[id]) {
        coupled_rows.push_back(int(i));
        normalized.emplace_back();
        for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k) {
          int j = mass.Columns()[k];
          if (free[j]) {
            if (!coupled[j]) {
              error_ = "Inconsistent symmetric mass coupling";
              return false;
            }
            normalized.back().push_back({faces[j], mass.Weights()[k] * scale[id] * scale[j]});
          }
        }
      }
      std::map<int, double> row;
      if (f.low >= 0)
        row[f.low] = -f.area * f.aperture;
      if (f.high >= 0)
        row[f.high] = f.area * f.aperture;
      for (auto [cell, weight] : row)
        transpose[i].push_back({cells[cell], weight});
      for (int k = mass.Offsets()[id]; k < mass.Offsets()[id + 1]; ++k)
        m[i].push_back({faces[mass.Columns()[k]], mass.Weights()[k]});
    }
    auto &rank = next->ranks[r];
    int device = devices[r];
    if (!next->MatrixBuild(device, rank.two_point, two_point) || !next->MatrixBuild(device, rank.d, d) ||
        !next->MatrixBuild(device, rank.transpose, transpose) || !next->MatrixBuild(device, rank.mass, m) ||
        !next->MatrixBuild(device, rank.normalized, normalized) ||
        !next->Upload(device, rank.coupled_rows, coupled_rows) || !next->Upload(device, rank.free, local_free) ||
        !next->Upload(device, rank.coupled, local_coupled) || !next->Upload(device, rank.scale, local_scale) ||
        !next->Upload(device, rank.diagonal, diagonal) ||
        !next->Allocate(device, rank.inverse_rhs, layout.owned_faces) ||
        !next->Allocate(device, rank.inverse_result, layout.owned_faces) ||
        !next->Allocate(device, rank.b, layout.owned_faces) ||
        !next->Allocate(device, rank.direction, layout.owned_faces) || !next->Upload(device, rank.active, active) ||
        !next->Allocate(device, rank.gradient, layout.faces.size()) ||
        !next->Allocate(device, rank.flux, layout.faces.size()) ||
        !next->Allocate(device, rank.x, layout.faces.size()) ||
        !next->Allocate(device, rank.scratch, layout.faces.size()) ||
        !next->Check(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking)) ||
        !next->Check(cudaEventCreateWithFlags(&rank.captured, cudaEventDisableTiming))) {
      error_ = next->error;
      return false;
    }
  }
  for (int device : devices)
    if (!next->Check(cudaSetDevice(device)) || !next->Check(cudaDeviceSynchronize())) {
      error_ = next->error;
      return false;
    }
  if (!(devices.size() > 1 ? next->BuildNeighborhoods(mass, partition, coupled, scale) : next->CaptureInverse())) {
    error_ = next->error;
    return false;
  }
  impl_ = std::move(next);
  return true;
}

const std::vector<PressureLocalLayout> &GpuPressureOperators::Layouts() const {
  static const std::vector<PressureLocalLayout> empty;
  return impl_ ? impl_->exchange.Layouts() : empty;
}

bool GpuPressureOperators::Apply(const std::vector<double *> &input,
                                 const std::vector<double *> &output,
                                 int operation) {
  error_.clear();
  if (!impl_ || input.size() != impl_->ranks.size() || output.size() != input.size() ||
      std::any_of(input.begin(), input.end(), [](auto p) { return !p; }) ||
      std::any_of(output.begin(), output.end(),
                  [&](auto p) { return !p || std::find(input.begin(), input.end(), p) != input.end(); })) {
    error_ = "Pressure operator needs disjoint input and output fields per rank";
    return false;
  }
  auto &w = *impl_;
  DeviceScope restore;
  if (!((operation == 1 || operation == 3) ? w.exchange.ExchangeCells(input) : w.exchange.ExchangeFaces(input))) {
    error_ = w.exchange.Error();
    return false;
  }
  bool ok = true;
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    if (!w.Check(cudaSetDevice(w.devices[r]))) {
      ok = false;
      break;
    }
    auto &rank = w.ranks[r];
    Matrix matrix = operation == 0   ? rank.d
                    : operation == 1 ? rank.transpose
                    : operation == 2 ? rank.mass
                                     : rank.two_point;
    Multiply<<<(matrix.rows + 255) / 256, 256, 0, rank.stream>>>(matrix, input[r], output[r]);
    if (!w.Check(cudaGetLastError())) {
      ok = false;
      break;
    }
  }
  std::string failure = ok ? std::string{} : w.error;
  for (size_t r = 0; r < w.ranks.size(); ++r)
    if (!w.Check(cudaSetDevice(w.devices[r])) || !w.Check(cudaStreamSynchronize(w.ranks[r].stream))) {
      if (failure.empty())
        failure = w.error;
      ok = false;
    }
  if (!ok)
    error_ = failure;
  return ok;
}

bool GpuPressureOperators::Divergence(const std::vector<double *> &faces, const std::vector<double *> &cells) {
  return Apply(faces, cells, 0);
}

bool GpuPressureOperators::TransposeDivergence(const std::vector<double *> &cells, const std::vector<double *> &faces) {
  return Apply(cells, faces, 1);
}

bool GpuPressureOperators::Mass(const std::vector<double *> &input, const std::vector<double *> &output) {
  return Apply(input, output, 2);
}

bool GpuPressureOperators::TwoPointPressure(const std::vector<double *> &input, const std::vector<double *> &output) {
  return Apply(input, output, 3);
}

bool GpuPressureOperators::InverseMass(const std::vector<double *> &input, const std::vector<double *> &output) {
  error_.clear();
  if (!impl_ || input.size() != impl_->ranks.size() || output.size() != input.size() ||
      std::any_of(input.begin(), input.end(), [](auto p) { return !p; }) ||
      std::any_of(output.begin(), output.end(),
                  [&](auto p) { return !p || std::find(input.begin(), input.end(), p) != input.end(); })) {
    error_ = "Mass inverse needs disjoint input and output fields per rank";
    return false;
  }
  DeviceScope restore;
  auto &w = *impl_;
  auto fail = [&] {
    std::string first = w.error;
    w.Synchronize();
    error_ = first;
    return false;
  };
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(rank.inverse_rhs, input[r], rank.mass.rows * sizeof(double), cudaMemcpyDeviceToDevice,
                                 rank.stream)))
      return fail();
  }
  // Every owning RHS is stable before any neighborhood graph reads peers.
  if (!w.Synchronize())
    return fail();
  if (w.devices.size() > 1) {
    for (size_t r = 0; r < w.ranks.size(); ++r)
      if (!w.Check(cudaSetDevice(w.devices[r])) ||
          !w.Check(cudaGraphLaunch(w.ranks[r].local_inverse, w.ranks[r].stream)))
        return fail();
  } else if (!w.Check(cudaSetDevice(w.devices.front())) ||
             !w.Check(cudaGraphLaunch(w.inverse_graph, w.ranks.front().stream)))
    return fail();
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    if (!w.Check(cudaSetDevice(w.devices[r])) ||
        !w.Check(cudaMemcpyAsync(output[r], rank.inverse_result, rank.mass.rows * sizeof(double),
                                 cudaMemcpyDeviceToDevice, rank.stream)))
      return fail();
  }
  if (!w.Synchronize())
    return fail();
  return true;
}

bool GpuPressureOperators::Gradient(const std::vector<double *> &cells, const std::vector<double *> &faces) {
  error_.clear();
  if (!impl_ || cells.size() != impl_->ranks.size() || faces.size() != cells.size() ||
      std::any_of(cells.begin(), cells.end(), [](auto p) { return !p; }) ||
      std::any_of(faces.begin(), faces.end(),
                  [&](auto p) { return !p || std::find(cells.begin(), cells.end(), p) != cells.end(); })) {
    error_ = "Gradient needs disjoint cell and face fields per rank";
    return false;
  }
  std::vector<double *> transpose;
  for (auto &rank : impl_->ranks)
    transpose.push_back(rank.gradient);
  return TransposeDivergence(cells, transpose) && InverseMass(transpose, faces);
}

bool GpuPressureOperators::Pressure(const std::vector<double *> &input, const std::vector<double *> &output) {
  error_.clear();
  if (!impl_ || input.size() != impl_->ranks.size() || output.size() != input.size() ||
      std::any_of(input.begin(), input.end(), [](auto p) { return !p; }) ||
      std::any_of(output.begin(), output.end(),
                  [&](auto p) { return !p || std::find(input.begin(), input.end(), p) != input.end(); })) {
    error_ = "Pressure needs disjoint cell fields per rank";
    return false;
  }
  DeviceScope restore;
  auto &w = *impl_;
  std::vector<double *> flux;
  for (auto &rank : w.ranks)
    flux.push_back(rank.flux);
  if (!Gradient(input, flux) || !Divergence(flux, output))
    return false;
  bool ok = true;
  for (size_t r = 0; r < w.ranks.size(); ++r) {
    auto &rank = w.ranks[r];
    if (!w.Check(cudaSetDevice(w.devices[r]))) {
      ok = false;
      break;
    }
    PinInactive<<<(rank.d.rows + 255) / 256, 256, 0, rank.stream>>>(rank.d.rows, rank.active, input[r], output[r]);
    if (!w.Check(cudaGetLastError())) {
      ok = false;
      break;
    }
  }
  std::string first = ok ? std::string{} : w.error;
  if (!w.Synchronize()) {
    if (first.empty())
      first = w.error;
    ok = false;
  }
  if (!ok)
    error_ = first;
  return ok;
}

}  // namespace wind3d::amr
