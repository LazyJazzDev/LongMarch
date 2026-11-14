#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <limits>
#include <map>

#include "amr_gpu_projection.h"

namespace wind3d::amr {
namespace {
struct Matrix {
  int rows{};
  int *offsets{}, *columns{};
  double *weights{};
};

__global__ void Multiply(Matrix a, const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.rows)
    return;
  double sum = 0;
  for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
    sum += a.weights[k] * x[a.columns[k]];
  y[i] = sum;
}

__global__ void
Search(int n, const double *r, double *p, const double *v, const double *d, double *phat, double beta, double omega) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  p[i] = r[i] + beta * (p[i] - omega * v[i]);
  phat[i] = p[i] / d[i];
}

__global__ void Advance(int n,
                        const double *r,
                        const double *v,
                        const double *d,
                        const double *phat,
                        double *s,
                        double *shat,
                        double *x,
                        double alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  s[i] = r[i] - alpha * v[i];
  shat[i] = s[i] / d[i];
  x[i] += alpha * phat[i];
}

__global__ void Add(int n, double *x, const double *y, double factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += factor * y[i];
}

__global__ void Difference(int n, const double *a, const double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

__global__ void CloseFaces(int count, const int *indices, double *velocity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    velocity[indices[i]] = 0;
}

using Rows = std::vector<std::map<int, double>>;
}  // namespace

struct GpuProjection::Impl {
  int device{}, n{}, nf{};
  bool ok{true};
  std::string error;
  cublasHandle_t blas{};
  std::vector<void *> allocations;
  Matrix a, g, d;
  std::vector<int> closed;
  int *closed_device{};
  double *diagonal{}, *u{}, *face{}, *rhs{}, *x{}, *r{}, *shadow{}, *p{}, *v{}, *s{}, *t{}, *phat{}, *shat{}, *ax{};

  ~Impl() {
    cudaSetDevice(device);
    if (blas)
      cublasDestroy(blas);
    for (void *ptr : allocations)
      cudaFree(ptr);
  }

  bool Check(cudaError_t code) {
    if (code != cudaSuccess) {
      ok = false;
      error = cudaGetErrorString(code);
    }
    return ok;
  }

  bool Blas(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS) {
      ok = false;
      error = "cuBLAS failure " + std::to_string(int(code));
    }
    return ok;
  }

  template <class T>
  bool Allocate(T *&ptr, size_t count) {
    if (!Check(cudaMalloc(&ptr, count * sizeof(T))))
      return false;
    allocations.push_back(ptr);
    return true;
  }

  template <class T>
  bool Upload(T *&ptr, const std::vector<T> &data) {
    return Allocate(ptr, data.size()) &&
           Check(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  bool UploadMatrix(Matrix &m, const Rows &rows) {
    std::vector<int> offsets{0}, columns;
    std::vector<double> weights;
    for (const auto &row : rows) {
      for (auto [col, w] : row)
        if (w != 0) {
          columns.push_back(col);
          weights.push_back(w);
        }
      if (weights.size() > size_t(INT_MAX))
        return false;
      offsets.push_back(int(weights.size()));
    }
    m.rows = int(rows.size());
    return Upload(m.offsets, offsets) && Upload(m.columns, columns) && Upload(m.weights, weights);
  }

  void Apply(Matrix m, const double *in, double *out) {
    Multiply<<<(m.rows + 255) / 256, 256>>>(m, in, out);
    Check(cudaGetLastError());
  }

  double Dot(const double *a, const double *b) {
    double result = 0;
    Blas(cublasDdot(blas, n, a, 1, b, 1, &result));
    return result;
  }
};

GpuProjection::GpuProjection() = default;
GpuProjection::~GpuProjection() = default;

std::string GpuProjection::Error() const {
  return !build_error_.empty() ? build_error_ : (impl_ ? impl_->error : "GPU projection is not built");
}

bool GpuProjection::Build(int device, const Mesh &mesh, const FaceGradient &gradient) {
  auto next = std::make_unique<Impl>();
  next->device = device;
  auto fail = [&](const char *reason) {
    build_error_ = next->error.empty() ? reason : next->error;
    return false;
  };
  if (!next->Check(cudaSetDevice(device)) || !next->Blas(cublasCreate(&next->blas)))
    return fail("Invalid adaptive operator or GPU allocation failure");
  if (mesh.Cells().empty() || mesh.Cells().size() > size_t(INT_MAX) || mesh.Faces().size() > size_t(INT_MAX) ||
      gradient.Offsets().size() != mesh.Faces().size() + 1)
    return fail("Invalid adaptive operator or GPU allocation failure");
  next->n = int(mesh.Cells().size());
  next->nf = int(mesh.Faces().size());
  Rows g(next->nf), d(next->n), a(next->n);
  std::vector<double> diagonal(next->n);
  for (int f = 0; f < next->nf; ++f) {
    for (int k = gradient.Offsets()[f]; k < gradient.Offsets()[f + 1]; ++k)
      g[f][gradient.Columns()[k]] += gradient.Weights()[k];
    auto face = mesh.Faces()[f];
    if (face.aperture == 0)
      next->closed.push_back(f);
    face.area *= face.aperture;
    if (face.aperture > 0 && face.axis == 0 && face.high < 0)
      g[f][face.low] = -1 / face.distance;
    if (face.low >= 0)
      d[face.low][f] = -face.area;
    if (face.high >= 0)
      d[face.high][f] = face.area;
  }
  for (int i = 0; i < next->n; ++i) {
    for (auto [f, area] : d[i])
      for (auto [col, w] : g[f])
        a[i][col] += area * w;
    if (!mesh.Cells()[i].active)
      a[i][i] = 1;
    diagonal[i] = a[i][i];
    if (!(diagonal[i] > 0) || !std::isfinite(diagonal[i]))
      return fail("Invalid adaptive operator or GPU allocation failure");
  }
  if (!next->UploadMatrix(next->a, a) || !next->UploadMatrix(next->g, g) || !next->UploadMatrix(next->d, d) ||
      !next->Upload(next->diagonal, diagonal))
    return fail("Invalid adaptive operator or GPU allocation failure");
  for (double **ptr : {&next->u, &next->face})
    if (!next->Allocate(*ptr, next->nf))
      return fail("Invalid adaptive operator or GPU allocation failure");
  for (double **ptr : {&next->rhs, &next->x, &next->r, &next->shadow, &next->p, &next->v, &next->s, &next->t,
                       &next->phat, &next->shat, &next->ax})
    if (!next->Allocate(*ptr, next->n))
      return fail("Invalid adaptive operator or GPU allocation failure");
  if (!next->closed.empty() && !next->Upload(next->closed_device, next->closed))
    return fail("Closed-face upload failed");
  impl_ = std::move(next);
  build_error_.clear();
  return true;
}

ProjectionResult GpuProjection::Project(std::vector<double> &velocity,
                                        std::vector<double> &potential,
                                        double tolerance,
                                        int max_iterations) {
  if (!impl_ || velocity.size() != size_t(impl_->nf) || !(tolerance > 0) || !std::isfinite(tolerance) ||
      max_iterations <= 0)
    return {};
  auto &w = *impl_;
  if (!w.Check(cudaSetDevice(w.device)) || !w.ok ||
      !w.Check(cudaMemcpy(w.u, velocity.data(), w.nf * sizeof(double), cudaMemcpyHostToDevice)))
    return {};
  auto result = Solve(tolerance, max_iterations);
  if (result.converged) {
    std::vector<double> new_u(w.nf), new_p(w.n);
    if (w.Check(cudaMemcpy(new_u.data(), w.u, w.nf * sizeof(double), cudaMemcpyDeviceToHost)) &&
        w.Check(cudaMemcpy(new_p.data(), w.x, w.n * sizeof(double), cudaMemcpyDeviceToHost))) {
      velocity = std::move(new_u);
      potential = std::move(new_p);
    } else
      result.converged = false;
  }
  return result;
}

ProjectionResult GpuProjection::ProjectDevice(double *velocity,
                                              size_t faces,
                                              double *potential,
                                              size_t cells,
                                              double tolerance,
                                              int max_iterations) {
  if (!impl_ || !velocity || !potential || velocity == potential || faces != size_t(impl_->nf) ||
      cells != size_t(impl_->n) || !(tolerance > 0) || !std::isfinite(tolerance) || max_iterations <= 0)
    return {};
  auto &w = *impl_;
  if (!w.Check(cudaSetDevice(w.device)) || !w.ok ||
      !w.Check(cudaMemcpy(w.u, velocity, faces * sizeof(double), cudaMemcpyDeviceToDevice)))
    return {};
  auto result = Solve(tolerance, max_iterations);
  if (result.converged && (!w.Check(cudaMemcpy(velocity, w.u, faces * sizeof(double), cudaMemcpyDeviceToDevice)) ||
                           !w.Check(cudaMemcpy(potential, w.x, cells * sizeof(double), cudaMemcpyDeviceToDevice))))
    result.converged = false;
  return result;
}

ProjectionResult GpuProjection::Solve(double tolerance, int max_iterations) {
  ProjectionResult result;
  auto &w = *impl_;
  auto copy = [&](double *dst, const double *src) {
    return w.Check(cudaMemcpy(dst, src, w.n * sizeof(double), cudaMemcpyDeviceToDevice));
  };
  for (double *ptr : {w.x, w.p, w.v})
    if (!w.Check(cudaMemset(ptr, 0, w.n * sizeof(double))))
      return result;
  w.Apply(w.d, w.u, w.rhs);
  double norm = std::sqrt(w.Dot(w.rhs, w.rhs));
  if (!w.ok || !std::isfinite(norm))
    return result;
  if (norm == 0) {
    if (!w.closed.empty())
      CloseFaces<<<(w.closed.size() + 255) / 256, 256>>>(int(w.closed.size()), w.closed_device, w.u);
    result.converged = w.Check(cudaGetLastError());
    return result;
  }
  copy(w.r, w.rhs);
  copy(w.shadow, w.rhs);
  double rho_old = 1, alpha = 1, omega = 1;
  result.relative_residual = 1;
  int blocks = (w.n + 255) / 256;
  for (int iteration = 0; iteration < max_iterations && w.ok; ++iteration) {
    result.iterations = iteration + 1;
    double rho = w.Dot(w.shadow, w.r);
    if (!std::isfinite(rho) || std::abs(rho) < 1e-300 || std::abs(omega) < 1e-300)
      break;
    double beta = (rho / rho_old) * (alpha / omega);
    Search<<<blocks, 256>>>(w.n, w.r, w.p, w.v, w.diagonal, w.phat, beta, omega);
    w.Apply(w.a, w.phat, w.v);
    double divisor = w.Dot(w.shadow, w.v);
    if (!std::isfinite(divisor) || std::abs(divisor) < 1e-300)
      break;
    alpha = rho / divisor;
    Advance<<<blocks, 256>>>(w.n, w.r, w.v, w.diagonal, w.phat, w.s, w.shat, w.x, alpha);
    if (std::sqrt(w.Dot(w.s, w.s)) / norm > tolerance) {
      w.Apply(w.a, w.shat, w.t);
      double tt = w.Dot(w.t, w.t);
      if (!std::isfinite(tt) || tt < 1e-300)
        break;
      omega = w.Dot(w.t, w.s) / tt;
      Add<<<blocks, 256>>>(w.n, w.x, w.shat, omega);
    }
    w.Apply(w.a, w.x, w.ax);
    Difference<<<blocks, 256>>>(w.n, w.rhs, w.ax, w.r);
    result.relative_residual = std::sqrt(w.Dot(w.r, w.r)) / norm;
    if (!std::isfinite(result.relative_residual))
      break;
    if (w.ok && result.relative_residual <= tolerance) {
      result.converged = true;
      break;
    }
    rho_old = rho;
  }
  if (result.converged) {
    w.Apply(w.g, w.x, w.face);
    Add<<<(w.nf + 255) / 256, 256>>>(w.nf, w.u, w.face, -1);
    if (!w.closed.empty())
      CloseFaces<<<(w.closed.size() + 255) / 256, 256>>>(int(w.closed.size()), w.closed_device, w.u);
    if (!w.Check(cudaGetLastError()))
      result.converged = false;
  }
  return result;
}
}  // namespace wind3d::amr
