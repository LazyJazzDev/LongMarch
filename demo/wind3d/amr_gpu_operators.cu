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
    Matrix d, transpose, mass;
    cudaStream_t stream{};
  };

  GpuPressureExchange exchange;
  std::vector<int> devices;
  std::vector<Rank> ranks;
  std::vector<std::pair<int, void *>> allocations;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t r = 0; r < ranks.size(); ++r)
      if (ranks[r].stream) {
        cudaSetDevice(devices[r]);
        cudaStreamSynchronize(ranks[r].stream);
        cudaStreamDestroy(ranks[r].stream);
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

bool GpuPressureOperators::Build(const Mesh &mesh, const Topology &tree, const std::vector<int> &devices) {
  error_.clear();
  DeviceScope restore;
  FaceMass mass;
  PressurePartition partition;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, int(devices.size()))) {
    error_ = "Invalid mesh or pressure partition";
    return false;
  }
  auto next = std::make_unique<Impl>();
  if (!next->exchange.Build(partition, devices)) {
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
    Rows d(layout.owned_cells), transpose(layout.owned_faces), m(layout.owned_faces);
    for (size_t i = 0; i < layout.owned_cells; ++i) {
      int cell = layout.cells[i];
      for (int k = mesh.Offsets()[cell]; k < mesh.Offsets()[cell + 1]; ++k) {
        int entry = mesh.Incidence()[k], id = std::abs(entry) - 1;
        const auto &f = mesh.Faces()[id];
        d[i].push_back({faces[id], (entry > 0 ? -1 : 1) * f.area * f.aperture});
      }
    }
    for (size_t i = 0; i < layout.owned_faces; ++i) {
      int id = layout.faces[i];
      const auto &f = mesh.Faces()[id];
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
    if (!next->MatrixBuild(device, rank.d, d) || !next->MatrixBuild(device, rank.transpose, transpose) ||
        !next->MatrixBuild(device, rank.mass, m) ||
        !next->Check(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking))) {
      error_ = next->error;
      return false;
    }
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
  if (!(operation == 1 ? w.exchange.ExchangeCells(input) : w.exchange.ExchangeFaces(input))) {
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
    Matrix matrix = operation == 0 ? rank.d : operation == 1 ? rank.transpose : rank.mass;
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
}  // namespace wind3d::amr
