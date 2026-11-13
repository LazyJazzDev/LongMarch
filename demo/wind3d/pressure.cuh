#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace wind3d::pressure {
struct Level {
  int nx{}, ny{}, nz{};
  float4 *edges{};  // +x,+y,+z conductances and outlet Dirichlet anchor.
  float *x{}, *rhs{}, *residual{};

  __host__ __device__ int Count() const {
    return nx * ny * nz;
  }

  __host__ __device__ int Index(int i, int j, int k) const {
    return (k * ny + j) * nx + i;
  }
};

__device__ float Diagonal(Level l, int n, int x, int y, int z) {
  float4 e = l.edges[n];
  return e.x + e.y + e.z + e.w + (x ? l.edges[n - 1].x : 0) + (y ? l.edges[n - l.nx].y : 0) +
         (z ? l.edges[n - l.nx * l.ny].z : 0);
}

__device__ float NeighborSum(Level l, int n, int x, int y, int z) {
  float4 e = l.edges[n];
  return (x + 1 < l.nx ? e.x * l.x[n + 1] : 0) + (y + 1 < l.ny ? e.y * l.x[n + l.nx] : 0) +
         (z + 1 < l.nz ? e.z * l.x[n + l.nx * l.ny] : 0) + (x ? l.edges[n - 1].x * l.x[n - 1] : 0) +
         (y ? l.edges[n - l.nx].y * l.x[n - l.nx] : 0) + (z ? l.edges[n - l.nx * l.ny].z * l.x[n - l.nx * l.ny] : 0);
}

__global__ void FineOperator(Level l,
                             const unsigned char *solid,
                             const unsigned char *bu,
                             const unsigned char *bv,
                             const unsigned char *bw) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= l.Count())
    return;
  int x = n % l.nx, y = n / l.nx % l.ny, z = n / (l.nx * l.ny);
  float4 e{};
  int face = (z * (l.ny + 1) + y) * (l.nx + 1) + x;
  if (!solid[n]) {
    e.x = x + 1 < l.nx && !solid[n + 1] && !bu[face + 1] ? 1 : 0;
    e.y = y + 1 < l.ny && !solid[n + l.nx] && !bv[face + l.nx + 1] ? 1 : 0;
    e.z = z + 1 < l.nz && !solid[n + l.nx * l.ny] && !bw[face + (l.nx + 1) * (l.ny + 1)] ? 1 : 0;
    e.w = x + 1 == l.nx ? 2 : 0;
  }
  l.edges[n] = e;
}

// Exact P^T A P for piecewise-constant 2x2x2 aggregates: internal edges cancel,
// crossing edges and pressure-boundary anchors add. This coarse operator is
// only a preconditioner; convergence is always measured on the fine operator.
__global__ void CoarseOperator(Level fine, Level coarse) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= coarse.Count())
    return;
  int x = 2 * (n % coarse.nx), y = 2 * (n / coarse.nx % coarse.ny), z = 2 * (n / (coarse.nx * coarse.ny));
  float4 sum{};
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        if (x + i >= fine.nx || y + j >= fine.ny || z + k >= fine.nz)
          continue;
        float4 e = fine.edges[fine.Index(x + i, y + j, z + k)];
        if (i)
          sum.x += e.x;
        if (j)
          sum.y += e.y;
        if (k)
          sum.z += e.z;
        sum.w += e.w;
      }
  coarse.edges[n] = sum;
}

__global__ void SetRhs(Level l, const float *divergence, float factor) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= l.Count())
    return;
  l.x[n] = 0;
  l.rhs[n] = -divergence[n] * factor;
}

__global__ void Smooth(Level l, int parity) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= l.Count())
    return;
  int x = n % l.nx, y = n / l.nx % l.ny, z = n / (l.nx * l.ny);
  if ((x + y + z) % 2 != parity)
    return;
  float d = Diagonal(l, n, x, y, z);
  if (d > 0)
    l.x[n] += 1.3f * ((l.rhs[n] + NeighborSum(l, n, x, y, z)) / d - l.x[n]);
}

__global__ void Residual(Level l) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= l.Count())
    return;
  int x = n % l.nx, y = n / l.nx % l.ny, z = n / (l.nx * l.ny);
  l.residual[n] = l.rhs[n] - Diagonal(l, n, x, y, z) * l.x[n] + NeighborSum(l, n, x, y, z);
}

__global__ void Restrict(Level fine, Level coarse) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= coarse.Count())
    return;
  int x = 2 * (n % coarse.nx), y = 2 * (n / coarse.nx % coarse.ny), z = 2 * (n / (coarse.nx * coarse.ny));
  float sum = 0;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        if (x + i < fine.nx && y + j < fine.ny && z + k < fine.nz)
          sum += fine.residual[fine.Index(x + i, y + j, z + k)];
  coarse.rhs[n] = sum;
  coarse.x[n] = 0;
}

__global__ void Prolong(Level coarse, Level fine) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= fine.Count())
    return;
  int x = n % fine.nx, y = n / fine.nx % fine.ny, z = n / (fine.nx * fine.ny);
  fine.x[n] += coarse.x[coarse.Index(x / 2, y / 2, z / 2)];
}

__global__ void DotProducts(int count, const float *a, const float *b, const float *c, const float *d, float2 *blocks) {
  __shared__ float2 values[256];
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  float2 v{};
  if (n < count)
    v = {a[n] * b[n], c[n] * d[n]};
  values[threadIdx.x] = v;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      values[threadIdx.x].x += values[threadIdx.x + offset].x;
      values[threadIdx.x].y += values[threadIdx.x + offset].y;
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    blocks[blockIdx.x] = values[0];
}

__global__ void Direction(int count, const float *z, float beta, float *d) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count)
    d[n] = z[n] + (beta == 0 ? 0 : beta * d[n]);
}

__global__ void Apply(Level l, float *result) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= l.Count())
    return;
  int x = n % l.nx, y = n / l.nx % l.ny, z = n / (l.nx * l.ny);
  result[n] = Diagonal(l, n, x, y, z) * l.x[n] - NeighborSum(l, n, x, y, z);
}

__global__ void Update(int count, float *x, const float *direction, float alpha) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count)
    x[n] += alpha * direction[n];
}

class Solver {
 public:
  Solver() = default;
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;

  ~Solver() {
    Release();
  }

  cudaError_t Initialize(int nx, int ny, int nz, cudaStream_t stream) {
    stream_ = stream;
    for (;;) {
      Level l{nx, ny, nz};
      cudaError_t error;
      if ((error = Allocate(l.edges, l.Count())) != cudaSuccess || (error = Allocate(l.x, l.Count())) != cudaSuccess ||
          (error = Allocate(l.rhs, l.Count())) != cudaSuccess ||
          (error = Allocate(l.residual, l.Count())) != cudaSuccess)
        return error;
      levels_.push_back(l);
      if (nx <= 4 && ny <= 4 && nz <= 4)
        break;
      nx = (nx + 1) / 2;
      ny = (ny + 1) / 2;
      nz = (nz + 1) / 2;
    }
    host_norm_.resize((levels_[0].Count() + 255) / 256);
    cudaError_t error;
    int count = levels_[0].Count();
    if ((error = Allocate(residual_, count)) != cudaSuccess || (error = Allocate(direction_, count)) != cudaSuccess ||
        (error = Allocate(product_, count)) != cudaSuccess)
      return error;
    if ((error = Allocate(norm_, int(host_norm_.size()))) != cudaSuccess)
      return error;
    if ((error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal)) != cudaSuccess)
      return error;
    cudaMemsetAsync(levels_[0].x, 0, size_t(count) * sizeof(float), stream_);
    float *rhs = levels_[0].rhs;
    levels_[0].rhs = residual_;
    Cycle(0);
    levels_[0].rhs = rhs;
    cudaGraph_t graph{};
    if ((error = cudaStreamEndCapture(stream_, &graph)) != cudaSuccess)
      return error;
    error = cudaGraphInstantiate(&preconditioner_, graph, 0);
    cudaGraphDestroy(graph);
    return error;
  }

  cudaError_t Build(const unsigned char *solid,
                    const unsigned char *bu,
                    const unsigned char *bv,
                    const unsigned char *bw) {
    auto first = levels_.front();
    FineOperator<<<Blocks(first), 256, 0, stream_>>>(first, solid, bu, bv, bw);
    for (size_t i = 1; i < levels_.size(); ++i)
      CoarseOperator<<<Blocks(levels_[i]), 256, 0, stream_>>>(levels_[i - 1], levels_[i]);
    return cudaGetLastError();
  }

  cudaError_t Solve(const float *divergence,
                    float *output,
                    float factor,
                    float tolerance,
                    int limit,
                    int &cycles,
                    float &relative_residual) {
    auto first = levels_.front();
    SetRhs<<<Blocks(first), 256, 0, stream_>>>(first, divergence, factor);
    cudaError_t error;
    // Warm-start from the previous physical pressure, but normalize convergence
    // against the new RHS and always measure the true b-Ax residual.
    Level initial = first;
    initial.x = output;
    initial.residual = residual_;
    Residual<<<Blocks(first), 256, 0, stream_>>>(initial);
    double rhs_norm = 0, previous_rho = 0, unused_norm;
    if ((error = Dot(first.rhs, first.rhs, first.rhs, first.rhs, rhs_norm, unused_norm)) != cudaSuccess)
      return error;
    relative_residual = 1;
    for (cycles = 0; cycles < limit;) {
      // A symmetric V-cycle is an SPD preconditioner. CG removes the slow
      // long-wavelength error left by unaccelerated aggregation multigrid.
      if ((error = cudaGraphLaunch(preconditioner_, stream_)) != cudaSuccess)
        return error;
      double rho, residual_norm;
      if ((error = Dot(residual_, first.x, residual_, residual_, rho, residual_norm)) != cudaSuccess)
        return error;
      relative_residual = float(std::sqrt(residual_norm / std::fmax(rhs_norm, 1e-30)));
      if (relative_residual <= tolerance)
        break;
      if (!(rho > 0) || !std::isfinite(rho))
        return cudaErrorUnknown;
      float beta = cycles ? float(rho / previous_rho) : 0;
      Direction<<<Blocks(first), 256, 0, stream_>>>(first.Count(), first.x, beta, direction_);
      Level operation = first;
      operation.x = direction_;
      Apply<<<Blocks(first), 256, 0, stream_>>>(operation, product_);
      double denominator, unused;
      if ((error = Dot(direction_, product_, direction_, product_, denominator, unused)) != cudaSuccess)
        return error;
      if (!(denominator > 0) || !std::isfinite(denominator))
        return cudaErrorUnknown;
      Update<<<Blocks(first), 256, 0, stream_>>>(first.Count(), output, direction_, float(rho / denominator));
      operation.x = output;
      operation.residual = residual_;
      // Recompute b-Ax, rather than accepting a drifted recursive CG residual.
      Residual<<<Blocks(first), 256, 0, stream_>>>(operation);
      if ((error = Dot(residual_, residual_, residual_, residual_, residual_norm, unused)) != cudaSuccess)
        return error;
      previous_rho = rho;
      ++cycles;
      relative_residual = float(std::sqrt(residual_norm / rhs_norm));
      if (!std::isfinite(relative_residual))
        return cudaErrorUnknown;
      if (relative_residual <= tolerance)
        break;
    }
    return cudaGetLastError();
  }

  void Release() {
    if (preconditioner_)
      cudaGraphExecDestroy(preconditioner_);
    preconditioner_ = nullptr;
    for (void *p : allocations_)
      cudaFree(p);
    allocations_.clear();
    levels_.clear();
  }

 private:
  static int Blocks(Level l) {
    return (l.Count() + 255) / 256;
  }

  template <class T>
  cudaError_t Allocate(T *&p, int count) {
    auto error = cudaMalloc(reinterpret_cast<void **>(&p), size_t(count) * sizeof(T));
    if (error == cudaSuccess)
      allocations_.push_back(p);
    return error;
  }

  void Cycle(size_t index) {
    Level l = levels_[index];
    int smooth = index + 1 == levels_.size() ? 16 : 3;
    for (int i = 0; i < smooth; ++i)
      for (int parity = 0; parity < 2; ++parity)
        Smooth<<<Blocks(l), 256, 0, stream_>>>(l, parity);
    if (index + 1 == levels_.size()) {
      for (int i = 0; i < smooth; ++i)
        for (int parity = 1; parity >= 0; --parity)
          Smooth<<<Blocks(l), 256, 0, stream_>>>(l, parity);
      return;
    }
    Residual<<<Blocks(l), 256, 0, stream_>>>(l);
    Level coarse = levels_[index + 1];
    Restrict<<<Blocks(coarse), 256, 0, stream_>>>(l, coarse);
    Cycle(index + 1);
    Prolong<<<Blocks(l), 256, 0, stream_>>>(coarse, l);
    for (int i = 0; i < smooth; ++i)
      for (int parity = 1; parity >= 0; --parity)
        Smooth<<<Blocks(l), 256, 0, stream_>>>(l, parity);
  }

  cudaError_t Dot(const float *a, const float *b, const float *c, const float *d, double &first, double &second) {
    DotProducts<<<Blocks(levels_[0]), 256, 0, stream_>>>(levels_[0].Count(), a, b, c, d, norm_);
    auto error =
        cudaMemcpyAsync(host_norm_.data(), norm_, host_norm_.size() * sizeof(float2), cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess)
      return error;
    if ((error = cudaStreamSynchronize(stream_)) != cudaSuccess)
      return error;
    first = second = 0;
    for (auto v : host_norm_) {
      first += v.x;
      second += v.y;
    }
    return cudaSuccess;
  }

  cudaGraphExec_t preconditioner_{};
  float *residual_{}, *direction_{}, *product_{};
  cudaStream_t stream_{};
  std::vector<Level> levels_;
  std::vector<void *> allocations_;
  float2 *norm_{};
  std::vector<float2> host_norm_;
};
}  // namespace wind3d::pressure
