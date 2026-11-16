#include <cuda_runtime.h>

#include <cmath>

#include "amr_gpu_mass.h"
#include "amr_mass.h"

namespace wind3d::amr {
namespace {
// Standard 2:1 split-cell patterns have spectrum [0.5,2] relative to
// diagonal face volumes. The enlarged interval absorbs coordinate roundoff.
constexpr double centre = 1.25, radius = .76;
constexpr int iterations = 32;

__global__ void DiagonalInverse(int n,
                                const unsigned char *free,
                                const double *diagonal,
                                const double *rhs,
                                double *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    result[i] = free[i] ? rhs[i] / diagonal[i] : 0;
}

__global__ void
Initialize(int n, const int *indices, const double *scale, const double *rhs, double *b, double *x, double *direction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    b[i] = rhs[indices[i]] * scale[i];
    x[i] = direction[i] = b[i] / centre;
  }
}

__global__ void Iterate(int n,
                        const int *offsets,
                        const int *columns,
                        const double *weights,
                        const double *b,
                        const double *x,
                        double *next,
                        double *direction,
                        double alpha,
                        double beta) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x, i = thread / 4;
  if (i >= n)
    return;
  double product = 0;
  for (int k = offsets[i] + threadIdx.x % 4; k < offsets[i + 1]; k += 4)
    product += weights[k] * x[columns[k]];
  unsigned mask = __activemask();
  product += __shfl_down_sync(mask, product, 2, 4);
  product += __shfl_down_sync(mask, product, 1, 4);
  if (threadIdx.x % 4 == 0) {
    double d = alpha * (b[i] - product) + beta * direction[i];
    next[i] = x[i] + d;
    direction[i] = d;
  }
}

__global__ void Scatter(int n, const int *indices, const double *scale, const double *x, double *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    result[indices[i]] = scale[i] * x[i];
}
}  // namespace

struct GpuMassInverse::Impl {
  int device{}, nf{}, coupled{};
  cudaStream_t stream{};
  cudaGraphExec_t graph{};
  std::vector<void *> allocations;
  unsigned char *free{};
  int *indices{}, *offsets{}, *columns{};
  double *diagonal{}, *scale{}, *weights{}, *rhs{}, *result{}, *b{}, *x{}, *scratch{}, *direction{};
  std::string error;

  ~Impl() {
    cudaSetDevice(device);
    if (graph)
      cudaGraphExecDestroy(graph);
    if (stream)
      cudaStreamDestroy(stream);
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
  bool Allocate(T *&p, size_t count) {
    if (!Check(cudaMalloc(&p, count * sizeof(T))))
      return false;
    allocations.push_back(p);
    return true;
  }

  template <class T>
  bool Upload(T *&p, const std::vector<T> &values) {
    return Allocate(p, values.size()) &&
           Check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  bool Capture() {
    if (!Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) ||
        !Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)))
      return false;
    DiagonalInverse<<<(nf + 255) / 256, 256, 0, stream>>>(nf, free, diagonal, rhs, result);
    if (coupled) {
      Initialize<<<(coupled + 255) / 256, 256, 0, stream>>>(coupled, indices, scale, rhs, b, x, direction);
      double rho = radius / centre;
      double *current = x, *next = scratch;
      for (int step = 1; step < iterations; ++step) {
        double updated = 1 / (2 * centre / radius - rho);
        Iterate<<<(coupled + 63) / 64, 256, 0, stream>>>(coupled, offsets, columns, weights, b, current, next,
                                                         direction, 2 * updated / radius, updated * rho);
        std::swap(current, next);
        rho = updated;
      }
      Scatter<<<(coupled + 255) / 256, 256, 0, stream>>>(coupled, indices, scale, current, result);
    }
    auto launch = cudaGetLastError();
    cudaGraph_t captured{};
    auto end = cudaStreamEndCapture(stream, &captured);
    if (!Check(launch) || !Check(end)) {
      if (captured)
        cudaGraphDestroy(captured);
      return false;
    }
    bool success = Check(cudaGraphInstantiate(&graph, captured, 0));
    cudaGraphDestroy(captured);
    return success;
  }
};

GpuMassInverse::GpuMassInverse() = default;
GpuMassInverse::~GpuMassInverse() = default;

bool GpuMassInverse::Build(int device, const Mesh &mesh, bool pressure_boundaries) {
  FaceMass mass;
  if (!mass.Build(mesh)) {
    error_ = "Unsupported compatible mass geometry";
    return false;
  }
  auto next = std::make_unique<Impl>();
  next->device = device;
  next->nf = int(mesh.Faces().size());
  auto fail = [&] {
    error_ = next->error;
    return false;
  };
  if (!next->Check(cudaSetDevice(device)))
    return fail();
  std::vector<unsigned char> free(next->nf);
  std::vector<int> mapping(next->nf, -1), indices, offsets{0}, columns;
  std::vector<double> scale, weights;
  for (int i = 0; i < next->nf; ++i) {
    const auto &f = mesh.Faces()[i];
    free[i] = f.aperture > 0 && (!pressure_boundaries || (f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0));
  }
  for (int i = 0; i < next->nf; ++i)
    if (free[i])
      for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
        if (mass.Columns()[k] != i && free[mass.Columns()[k]]) {
          mapping[i] = int(indices.size());
          indices.push_back(i);
          const auto &f = mesh.Faces()[i];
          scale.push_back(1 / std::sqrt(f.area * f.distance * f.aperture));
          break;
        }
  next->coupled = int(indices.size());
  for (int i : indices) {
    for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k) {
      int j = mass.Columns()[k];
      if (free[j]) {
        if (mapping[j] < 0) {
          error_ = "Inconsistent symmetric mass coupling";
          return false;
        }
        columns.push_back(mapping[j]);
        weights.push_back(mass.Weights()[k] * scale[mapping[i]] * scale[mapping[j]]);
      }
    }
    offsets.push_back(int(weights.size()));
  }
  if (!next->Upload(next->free, free) || !next->Upload(next->diagonal, mass.Diagonal()) ||
      !next->Allocate(next->rhs, next->nf) || !next->Allocate(next->result, next->nf))
    return fail();
  if (next->coupled && (!next->Upload(next->indices, indices) || !next->Upload(next->offsets, offsets) ||
                        !next->Upload(next->columns, columns) || !next->Upload(next->scale, scale) ||
                        !next->Upload(next->weights, weights) || !next->Allocate(next->b, next->coupled) ||
                        !next->Allocate(next->x, next->coupled) || !next->Allocate(next->scratch, next->coupled) ||
                        !next->Allocate(next->direction, next->coupled)))
    return fail();
  if (!next->Capture())
    return fail();
  impl_ = std::move(next);
  error_.clear();
  return true;
}

bool GpuMassInverse::ApplyDevice(const double *rhs, double *result, size_t faces) {
  if (!impl_ || !rhs || !result || rhs == result || faces != size_t(impl_->nf)) {
    error_ = "Invalid mass-inverse buffers";
    return false;
  }
  auto &w = *impl_;
  size_t bytes = faces * sizeof(double);
  if (!w.Check(cudaSetDevice(w.device)) || !w.Check(cudaMemcpyAsync(w.rhs, rhs, bytes, cudaMemcpyDeviceToDevice)) ||
      !w.Check(cudaGraphLaunch(w.graph, nullptr)) ||
      !w.Check(cudaMemcpyAsync(result, w.result, bytes, cudaMemcpyDeviceToDevice))) {
    error_ = w.error;
    return false;
  }
  error_.clear();
  return true;
}

size_t GpuMassInverse::CoupledFaces() const {
  return impl_ ? size_t(impl_->coupled) : 0;
}

std::string GpuMassInverse::Error() const {
  return error_;
}
}  // namespace wind3d::amr
