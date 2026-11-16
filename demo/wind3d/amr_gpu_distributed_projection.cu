#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "amr_gpu_distributed_projection.h"

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

__global__ void Initialize(int n, const double *rhs, double *x, double *r, double *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = p[i] = 0;
    r[i] = rhs[i];
  }
}

__global__ void Precondition(int n, const double *r, const double *diagonal, double *z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    z[i] = r[i] / diagonal[i];
}

__global__ void Direction(int n, const double *z, double *p, double beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = z[i] + beta * p[i];
}

__global__ void Advance(int n, double *x, double *r, const double *p, const double *q, double alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += alpha * p[i];
    r[i] -= alpha * q[i];
  }
}

__global__ void Residual(int n, const double *rhs, const double *q, double *r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    r[i] = rhs[i] - q[i];
}

__global__ void Correct(int n, const unsigned char *closed, double *u, const double *gradient, bool nonzero) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    u[i] = closed[i] ? 0 : u[i] - (nonzero ? gradient[i] : 0);
}
}  // namespace

struct GpuDistributedProjection::Impl {
  struct Rank {
    int cells{}, faces{};
    cudaStream_t stream{};
    cublasHandle_t blas{};
    double *u{}, *gradient{}, *rhs{}, *x{}, *r{}, *z{}, *p{}, *q{}, *diagonal{};
    unsigned char *closed{};
  };

  GpuPressureOperators operators;
  std::vector<int> devices;
  std::vector<Rank> ranks;
  std::vector<std::pair<int, void *>> allocations;
  std::string error;

  ~Impl() {
    DeviceScope restore;
    for (size_t i = 0; i < ranks.size(); ++i) {
      auto &r = ranks[i];
      if (r.stream || r.blas) {
        cudaSetDevice(devices[i]);
        if (r.stream)
          cudaStreamSynchronize(r.stream);
        if (r.blas)
          cublasDestroy(r.blas);
        if (r.stream)
          cudaStreamDestroy(r.stream);
      }
    }
    for (auto [d, p] : allocations) {
      cudaSetDevice(d);
      cudaFree(p);
    }
  }

  bool Check(cudaError_t status) {
    if (status == cudaSuccess)
      return true;
    error = cudaGetErrorString(status);
    return false;
  }

  bool Blas(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS)
      return true;
    error = "Distributed cuBLAS error " + std::to_string(int(status));
    return false;
  }

  template <class T>
  bool Allocate(int d, T *&p, size_t n) {
    if (!Check(cudaSetDevice(d)) || !Check(cudaMalloc(&p, n * sizeof(T))))
      return false;
    allocations.push_back({d, p});
    return true;
  }

  template <class T>
  bool Upload(int d, T *&p, const std::vector<T> &v) {
    return Allocate(d, p, v.size()) && Check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  std::vector<double *> Fields(double *Rank::*member) {
    std::vector<double *> v;
    for (auto &r : ranks)
      v.push_back(r.*member);
    return v;
  }

  bool Sync() {
    std::string first;
    for (size_t i = 0; i < ranks.size(); ++i)
      if (!Check(cudaSetDevice(devices[i])) || !Check(cudaStreamSynchronize(ranks[i].stream)))
        if (first.empty())
          first = error;
    if (first.empty())
      return true;
    error = first;
    return false;
  }

  bool Dot(double *Rank::*a, double *Rank::*b, double &result) {
    result = 0;
    double correction = 0;
    for (size_t i = 0; i < ranks.size(); ++i) {
      auto &r = ranks[i];
      double part = 0;
      if (!Check(cudaSetDevice(devices[i])) || !Blas(cublasDdot(r.blas, r.cells, r.*a, 1, r.*b, 1, &part)))
        return false;
      if (!std::isfinite(part)) {
        error = "Nonfinite distributed dot product";
        return false;
      }
      double adjusted = part - correction, next = result + adjusted;
      correction = (next - result) - adjusted;
      result = next;
    }
    if (!std::isfinite(result)) {
      error = "Nonfinite global dot product";
      return false;
    }
    return true;
  }

  bool Pressure(double *Rank::*input) {
    if (operators.Pressure(Fields(input), Fields(&Rank::q)))
      return true;
    error = operators.Error();
    return false;
  }

  bool TrueResidual(double norm, double &relative) {
    if (!Pressure(&Rank::x))
      return false;
    for (size_t i = 0; i < ranks.size(); ++i) {
      auto &r = ranks[i];
      if (!Check(cudaSetDevice(devices[i])))
        return false;
      Residual<<<(r.cells + 255) / 256, 256, 0, r.stream>>>(r.cells, r.rhs, r.q, r.r);
      if (!Check(cudaGetLastError()))
        return false;
    }
    double square = 0;
    if (!Dot(&Rank::r, &Rank::r, square))
      return false;
    relative = std::sqrt(square) / norm;
    if (!std::isfinite(relative)) {
      error = "Nonfinite true pressure residual";
      return false;
    }
    return true;
  }
};

GpuDistributedProjection::GpuDistributedProjection() = default;
GpuDistributedProjection::~GpuDistributedProjection() = default;

bool GpuDistributedProjection::Build(const Mesh &mesh, const Topology &tree, const std::vector<int> &devices) {
  error_.clear();
  DeviceScope restore;
  auto next = std::make_unique<Impl>();
  if (!next->operators.Build(mesh, tree, devices)) {
    error_ = next->operators.Error();
    return false;
  }
  next->devices = devices;
  next->ranks.resize(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    auto &r = next->ranks[i];
    const auto &layout = next->operators.Layouts()[i];
    int d = devices[i];
    r.cells = int(layout.owned_cells);
    r.faces = int(layout.owned_faces);
    std::vector<double> diagonal;
    std::vector<unsigned char> closed;
    for (size_t j = 0; j < layout.owned_cells; ++j) {
      int id = layout.cells[j];
      double value = 0;
      for (int k = mesh.Offsets()[id]; k < mesh.Offsets()[id + 1]; ++k) {
        auto f = mesh.Faces()[std::abs(mesh.Incidence()[k]) - 1];
        if ((f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0))
          value += f.area * f.aperture / f.distance;
      }
      if (!mesh.Cells()[id].active)
        value = 1;
      if (!(value > 0) || !std::isfinite(value)) {
        error_ = "Invalid distributed pressure diagonal";
        return false;
      }
      diagonal.push_back(value);
    }
    for (size_t j = 0; j < layout.owned_faces; ++j)
      closed.push_back(mesh.Faces()[layout.faces[j]].aperture == 0);
    if (!next->Upload(d, r.diagonal, diagonal) || !next->Upload(d, r.closed, closed) ||
        !next->Allocate(d, r.u, layout.faces.size()) || !next->Allocate(d, r.gradient, layout.faces.size())) {
      error_ = next->error;
      return false;
    }
    for (double **p : {&r.rhs, &r.x, &r.r, &r.z, &r.p, &r.q})
      if (!next->Allocate(d, *p, layout.cells.size())) {
        error_ = next->error;
        return false;
      }
    if (!next->Check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking)) ||
        !next->Blas(cublasCreate(&r.blas)) || !next->Blas(cublasSetStream(r.blas, r.stream))) {
      error_ = next->error;
      return false;
    }
  }
  impl_ = std::move(next);
  return true;
}

const std::vector<PressureLocalLayout> &GpuDistributedProjection::Layouts() const {
  static const std::vector<PressureLocalLayout> empty;
  return impl_ ? impl_->operators.Layouts() : empty;
}

ProjectionResult GpuDistributedProjection::Project(const std::vector<double *> &velocity,
                                                   const std::vector<double *> &potential,
                                                   double tolerance,
                                                   int max_iterations) {
  error_.clear();
  ProjectionResult result;
  if (!impl_ || velocity.size() != impl_->ranks.size() || potential.size() != velocity.size() || !(tolerance > 0) ||
      !std::isfinite(tolerance) || max_iterations <= 0 ||
      std::any_of(velocity.begin(), velocity.end(), [](auto p) { return !p; }) ||
      std::any_of(potential.begin(), potential.end(),
                  [&](auto p) { return !p || std::find(velocity.begin(), velocity.end(), p) != velocity.end(); })) {
    error_ = "Invalid distributed projection fields or tolerance";
    return result;
  }
  DeviceScope restore;
  auto &w = *impl_;
  w.error.clear();
  auto fail = [&] {
    std::string first = w.error;
    w.Sync();
    error_ = first;
    result.converged = false;
    return result;
  };
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    auto &r = w.ranks[i];
    if (!w.Check(cudaSetDevice(w.devices[i])) ||
        !w.Check(cudaMemcpyAsync(r.u, velocity[i], r.faces * sizeof(double), cudaMemcpyDeviceToDevice, r.stream)))
      return fail();
  }
  if (!w.Sync())
    return fail();
  if (!w.operators.Divergence(w.Fields(&Impl::Rank::u), w.Fields(&Impl::Rank::rhs))) {
    w.error = w.operators.Error();
    return fail();
  }
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    auto &r = w.ranks[i];
    if (!w.Check(cudaSetDevice(w.devices[i])))
      return fail();
    Initialize<<<(r.cells + 255) / 256, 256, 0, r.stream>>>(r.cells, r.rhs, r.x, r.r, r.p);
    if (!w.Check(cudaGetLastError()))
      return fail();
  }
  double square = 0;
  if (!w.Dot(&Impl::Rank::r, &Impl::Rank::r, square))
    return fail();
  double norm = std::sqrt(square);
  if (norm == 0)
    result.converged = true;
  else {
    double previous = 0;
    result.relative_residual = 1;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      result.iterations = iteration + 1;
      for (size_t i = 0; i < w.ranks.size(); ++i) {
        auto &r = w.ranks[i];
        if (!w.Check(cudaSetDevice(w.devices[i])))
          return fail();
        Precondition<<<(r.cells + 255) / 256, 256, 0, r.stream>>>(r.cells, r.r, r.diagonal, r.z);
        if (!w.Check(cudaGetLastError()))
          return fail();
      }
      double rho = 0;
      if (!w.Dot(&Impl::Rank::r, &Impl::Rank::z, rho))
        return fail();
      if (!(rho > 0)) {
        w.error = "Nonpositive distributed CG residual energy";
        break;
      }
      for (size_t i = 0; i < w.ranks.size(); ++i) {
        auto &r = w.ranks[i];
        if (!w.Check(cudaSetDevice(w.devices[i])))
          return fail();
        Direction<<<(r.cells + 255) / 256, 256, 0, r.stream>>>(r.cells, r.z, r.p, previous > 0 ? rho / previous : 0);
        if (!w.Check(cudaGetLastError()))
          return fail();
      }
      if (!w.Sync() || !w.Pressure(&Impl::Rank::p))
        return fail();
      double denominator = 0;
      if (!w.Dot(&Impl::Rank::p, &Impl::Rank::q, denominator))
        return fail();
      if (!(denominator > 0)) {
        w.error = "Nonpositive distributed CG curvature";
        break;
      }
      double alpha = rho / denominator;
      if (!std::isfinite(alpha)) {
        w.error = "Nonfinite distributed CG step";
        break;
      }
      for (size_t i = 0; i < w.ranks.size(); ++i) {
        auto &r = w.ranks[i];
        if (!w.Check(cudaSetDevice(w.devices[i])))
          return fail();
        Advance<<<(r.cells + 255) / 256, 256, 0, r.stream>>>(r.cells, r.x, r.r, r.p, r.q, alpha);
        if (!w.Check(cudaGetLastError()))
          return fail();
      }
      if (!w.Dot(&Impl::Rank::r, &Impl::Rank::r, square))
        return fail();
      double recursive = std::sqrt(square) / norm;
      previous = rho;
      if (recursive <= tolerance || (iteration + 1) % 50 == 0 || iteration + 1 == max_iterations) {
        if (!w.TrueResidual(norm, result.relative_residual))
          return fail();
        if (result.relative_residual <= tolerance) {
          result.converged = true;
          break;
        }
        previous = 0;
      }
    }
    if (!result.converged) {
      if (!w.TrueResidual(norm, result.relative_residual))
        return fail();
      if (w.error.empty())
        w.error = "Distributed pressure did not converge";
      return fail();
    }
  }
  if (norm > 0 && !w.operators.Gradient(w.Fields(&Impl::Rank::x), w.Fields(&Impl::Rank::gradient))) {
    w.error = w.operators.Error();
    return fail();
  }
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    auto &r = w.ranks[i];
    if (!w.Check(cudaSetDevice(w.devices[i])))
      return fail();
    Correct<<<(r.faces + 255) / 256, 256, 0, r.stream>>>(r.faces, r.closed, r.u, r.gradient, norm > 0);
    if (!w.Check(cudaGetLastError()))
      return fail();
  }
  if (!w.Sync())
    return fail();
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    auto &r = w.ranks[i];
    if (!w.Check(cudaSetDevice(w.devices[i])) ||
        !w.Check(cudaMemcpyAsync(velocity[i], r.u, r.faces * sizeof(double), cudaMemcpyDeviceToDevice, r.stream)) ||
        !w.Check(cudaMemcpyAsync(potential[i], r.x, r.cells * sizeof(double), cudaMemcpyDeviceToDevice, r.stream)))
      return fail();
  }
  if (!w.Sync())
    return fail();
  return result;
}
}  // namespace wind3d::amr
