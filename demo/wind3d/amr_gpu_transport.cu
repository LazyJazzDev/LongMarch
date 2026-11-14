#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cuda/std/array>
#include <vector>

#include "amr_gpu_projection.h"
#include "amr_gpu_transport.h"
#include "wall_model.h"

namespace wind3d::amr {
namespace {
template <class T, size_t N>
using Array = cuda::std::array<T, N>;

__device__ double Min(double a, double b) {
  return a < b ? a : b;
}

__device__ double Max(double a, double b) {
  return a > b ? a : b;
}

__device__ double Clamp(double a, double lo, double hi) {
  return Min(Max(a, lo), hi);
}

struct Cell {
  Array<double, 3> centre;
  double width;
  bool active;
};

struct Face {
  int low, high, axis;
  Array<double, 3> centre;
  double area, aperture;
};

__device__ double Cubic(double f0, double f1, double f2, double f3, double t) {
  double delta = f2 - f1, d1 = .5 * (f2 - f0), d2 = .5 * (f3 - f1);
  if (delta == 0 || d1 * delta <= 0)
    d1 = 0;
  if (delta == 0 || d2 * delta <= 0)
    d2 = 0;
  double t2 = t * t, t3 = t2 * t;
  return Clamp((2 * t3 - 3 * t2 + 1) * f1 + (t3 - 2 * t2 + t) * d1 + (-2 * t3 + 3 * t2) * f2 + (t3 - t2) * d2,
               Min(f1, f2), Max(f1, f2));
}

struct Field {
  int n, max_level, nb, nc, nf;
  Array<int, 3> roots;
  Array<double, 3> origin_;
  double root_width_;
  const Block *leaves;
  const Cell *cells;
  const Face *faces;
  const int *offsets, *incidence;
  double *velocity_;
  Array<double, 3> *centres_;
  Array<double, 9> *slopes_;
  Array<double, 6> *limits_;

  __device__ int Find(int x, int y, int z) const {
    for (int level = max_level; level >= 0; --level) {
      int scale = 1 << (max_level - level), qx = x / scale, qy = y / scale, qz = z / scale;
      int lo = 0, hi = nb;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        Block b = leaves[mid];
        bool less =
            b.level < level || (b.level == level && (b.z < qz || (b.z == qz && (b.y < qy || (b.y == qy && b.x < qx)))));
        if (less)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo < nb) {
        Block b = leaves[lo];
        if (b.level == level && b.x == qx && b.y == qy && b.z == qz)
          return lo;
      }
    }
    return -1;
  }

  __device__ int CellAt(Array<double, 3> point) const {
    const int n = this->n;
    int fine[3];
    for (int a = 0; a < 3; ++a) {
      double q = (point[a] - origin_[a]) / root_width_;
      q = Clamp(q, 0.0, nextafter(double(roots[a]), 0.0));
      fine[a] = int(q * n * (1 << max_level));
    }
    int block = Find(fine[0] / n, fine[1] / n, fine[2] / n);
    if (block < 0)
      return -1;
    auto b = leaves[block];
    int scale = 1 << (max_level - b.level);
    int x = fine[0] / scale - b.x * n, y = fine[1] / scale - b.y * n, z = fine[2] / scale - b.z * n;
    return block * n * n * n + (z * n + y) * n + x;
  }

  __device__ int NativeFace(Array<double, 3> point, int axis) const {
    int cell = CellAt(point);
    if (cell < 0)
      return -1;
    const auto &c = cells[cell];
    for (int k = offsets[cell]; k < offsets[cell + 1]; ++k) {
      int f = abs(incidence[k]) - 1;
      const auto &face = faces[f];
      if (face.axis != axis)
        continue;
      double error = 0;
      for (int a = 0; a < 3; ++a)
        error = Max(error, fabs(point[a] - face.centre[a]));
      if (error < c.width * 1e-9)
        return f;
    }
    return -1;
  }

  __device__ double VirtualFace(Array<double, 3> point, int axis) const {
    int cell = CellAt(point);
    if (cell < 0)
      return 0;
    int native = NativeFace(point, axis);
    if (native >= 0)
      return velocity_[native];
    const auto &c = cells[cell];
    double value = centres_[cell][axis];
    for (int a = 0; a < 3; ++a)
      value += slopes_[cell][3 * axis + a] * (point[a] - c.centre[a]);
    return Clamp(value, limits_[cell][2 * axis], limits_[cell][2 * axis + 1]);
  }

  __device__ double Component(Array<double, 3> point, int axis, bool cubic) const {
    for (int a = 0; a < 3; ++a) {
      if (!isfinite(point[a]))
        return nan("");
      point[a] = Clamp(point[a], origin_[a], origin_[a] + roots[a] * root_width_);
    }

    int cell = CellAt(point);
    if (cell < 0)
      return 0;
    double h = cells[cell].width, base[3], fraction[3];
    for (int a = 0; a < 3; ++a) {
      double q = (point[a] - origin_[a]) / h - (a == axis ? 0 : .5);
      base[a] = floor(q);
      fraction[a] = q - base[a];
    }
    auto corner_point = [&](int x, int y, int z) {
      int corner[3]{x, y, z};
      Array<double, 3> p{};
      for (int a = 0; a < 3; ++a) {
        p[a] = origin_[a] + (base[a] + corner[a] + (a == axis ? 0 : .5)) * h;
        p[a] = Clamp(p[a], origin_[a] + (a == axis ? 0 : .5 * h),
                     origin_[a] + roots[a] * root_width_ - (a == axis ? 0 : .5 * h));
      }
      return p;
    };
    if (cubic) {
      for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
          for (int x = 0; x < 2; ++x) {
            int f = NativeFace(corner_point(x, y, z), axis);
            if (f < 0 || faces[f].aperture != 1)
              cubic = false;
          }
      if (cubic) {
        double planes[4];
        for (int z = 0; z < 4; ++z) {
          double rows[4];
          for (int y = 0; y < 4; ++y) {
            double values[4];
            for (int x = 0; x < 4; ++x)
              values[x] = VirtualFace(corner_point(x - 1, y - 1, z - 1), axis);
            rows[y] = Cubic(values[0], values[1], values[2], values[3], fraction[0]);
          }
          planes[z] = Cubic(rows[0], rows[1], rows[2], rows[3], fraction[1]);
        }
        return Cubic(planes[0], planes[1], planes[2], planes[3], fraction[2]);
      }
    }
    double result = 0, weighted = 0, total = 0;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          int corner[3]{x, y, z};
          Array<double, 3> p{};
          double weight = 1;
          for (int a = 0; a < 3; ++a) {
            p[a] = origin_[a] + (base[a] + corner[a] + (a == axis ? 0 : .5)) * h;
            weight *= corner[a] ? fraction[a] : 1 - fraction[a];
            double lower = origin_[a] + (a == axis ? 0 : .5 * h),
                   upper = origin_[a] + roots[a] * root_width_ - (a == axis ? 0 : .5 * h);
            p[a] = Clamp(p[a], lower, upper);
          }
          double value = VirtualFace(p, axis);
          result += weight * value;
          int f = NativeFace(p, axis);
          double aperture = f < 0 ? 1 : faces[f].aperture;
          weighted += aperture * weight * value;
          total += aperture * weight;
        }
    return total > 1e-3 ? weighted / total : result;
  }

  __device__ Array<double, 3> Sample(Array<double, 3> p) const {
    return {Component(p, 0, false), Component(p, 1, false), Component(p, 2, false)};
  }

  __device__ Array<double, 3> Departure(Array<double, 3> p, double dt) const {
    auto u = Sample(p), mid = p;
    for (int a = 0; a < 3; ++a)
      mid[a] -= .5 * dt * u[a];
    u = Sample(mid);
    for (int a = 0; a < 3; ++a)
      p[a] -= dt * u[a];
    return p;
  }
};

__global__ void Centres(Field f) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= f.nc)
    return;
  Array<double, 3> value{}, weight{};
  for (int k = f.offsets[cell]; k < f.offsets[cell + 1]; ++k) {
    int face = abs(f.incidence[k]) - 1;
    auto g = f.faces[face];
    double area = g.area * g.aperture;
    weight[g.axis] += area;
    value[g.axis] += area * f.velocity_[face];
  }
  for (int a = 0; a < 3; ++a)
    if (weight[a] > 0)
      value[a] /= weight[a];
  f.centres_[cell] = value;
}

__global__ void Slopes(Field f) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= f.nc)
    return;
  auto centres = f.centres_;
  auto slopes = f.slopes_;
  auto limits = f.limits_;
  slopes[cell] = {};
  for (int a = 0; a < 3; ++a)
    limits[cell][2 * a] = limits[cell][2 * a + 1] = centres[cell][a];
  double matrix[3][6]{};
  for (int k = f.offsets[cell]; k < f.offsets[cell + 1]; ++k) {
    int face_id = abs(f.incidence[k]) - 1;
    auto face = f.faces[face_id];
    int other = face.low == int(cell) ? face.high : face.low;
    if (other < 0 || face.aperture == 0)
      continue;
    double d[3], r2 = 0;
    for (int a = 0; a < 3; ++a) {
      d[a] = f.cells[other].centre[a] - f.cells[cell].centre[a];
      r2 += d[a] * d[a];
      limits[cell][2 * a] = Min(limits[cell][2 * a], centres[other][a]);
      limits[cell][2 * a + 1] = Max(limits[cell][2 * a + 1], centres[other][a]);
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        matrix[a][b] += d[a] * d[b] / r2;
        matrix[a][3 + b] += d[a] * (centres[other][b] - centres[cell][b]) / r2;
      }
  }
  bool nonsingular = true;
  for (int a = 0; a < 3; ++a) {
    int pivot = a;
    for (int b = a + 1; b < 3; ++b)
      if (fabs(matrix[b][a]) > fabs(matrix[pivot][a]))
        pivot = b;
    if (fabs(matrix[pivot][a]) < 1e-12) {
      nonsingular = false;
      break;
    }
    for (int k = 0; k < 6; ++k) {
      double t = matrix[a][k];
      matrix[a][k] = matrix[pivot][k];
      matrix[pivot][k] = t;
    }
    double scale = matrix[a][a];
    for (int k = 0; k < 6; ++k)
      matrix[a][k] /= scale;
    for (int b = 0; b < 3; ++b)
      if (b != a) {
        double factor = matrix[b][a];
        for (int k = 0; k < 6; ++k)
          matrix[b][k] -= factor * matrix[a][k];
      }
  }
  if (nonsingular)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        slopes[cell][3 * b + a] = matrix[a][3 + b];
}

// Classify solid ghost layers once for each uploaded geometry.
__global__ void SlipNeighbors(Field f, int *neighbors, int *layer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.nf)
    return;
  layer[i] = 0;
  if (f.faces[i].aperture > 0)
    return;
  double h = sqrt(f.faces[i].area);
  for (int k = 0; k < 6; ++k) {
    auto p = f.faces[i].centre;
    p[k / 2] += (k % 2 ? 1 : -1) * h;
    int j = f.NativeFace(p, f.faces[i].axis);
    neighbors[6 * i + k] = j;
    if (j >= 0 && f.faces[j].aperture > 0)
      layer[i] = 1;
  }
}

__global__ void ClearSolid(Field f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < f.nf && f.faces[i].aperture == 0)
    f.velocity_[i] = 0;
}

__global__ void Extend(Field f, const int *neighbors, const int *layers, int pass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.nf || f.faces[i].aperture > 0 || (pass == 1 ? layers[i] != 1 : layers[i] == 1))
    return;
  double sum = 0;
  int count = 0;
  for (int k = 0; k < 6; ++k) {
    int j = neighbors[6 * i + k];
    if (j >= 0 && (pass == 1 ? f.faces[j].aperture > 0 : f.faces[j].aperture == 0 && layers[j] == 1)) {
      sum += f.velocity_[j];
      ++count;
    }
  }
  if (count)
    f.velocity_[i] = sum / count;
}

__global__ void Forward(Field f, double dt, bool cubic, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.nf)
    return;
  auto face = f.faces[i];
  out[i] = face.aperture == 0 ? 0 : f.Component(f.Departure(face.centre, dt), face.axis, cubic);
}

__global__ void Correct(Field original, Field forward, double dt, bool cubic, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= original.nf)
    return;
  auto face = original.faces[i];
  if (face.aperture == 0) {
    out[i] = 0;
    return;
  }
  double value =
      forward.velocity_[i] +
      .5 * (original.velocity_[i] - forward.Component(original.Departure(face.centre, -dt), face.axis, cubic));
  if (!isfinite(value)) {
    out[i] = nan("");
    return;
  }
  auto back = original.Departure(face.centre, dt);
  for (int a = 0; a < 3; ++a)
    back[a] = Clamp(back[a], original.origin_[a], original.origin_[a] + original.roots[a] * original.root_width_);
  int cell = original.CellAt(back);
  double h = original.cells[cell].width, base[3];
  for (int a = 0; a < 3; ++a)
    base[a] = floor((back[a] - original.origin_[a]) / h - (a == face.axis ? 0 : .5));
  double lo = INFINITY, hi = -INFINITY;
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        int corner[3]{x, y, z};
        Array<double, 3> p;
        for (int a = 0; a < 3; ++a)
          p[a] = Clamp(original.origin_[a] + (base[a] + corner[a] + (a == face.axis ? 0 : .5)) * h,
                       original.origin_[a] + (a == face.axis ? 0 : .5 * h),
                       original.origin_[a] + original.roots[a] * original.root_width_ - (a == face.axis ? 0 : .5 * h));
        double v = original.VirtualFace(p, face.axis);
        lo = Min(lo, v);
        hi = Max(hi, v);
      }
  out[i] = (value >= lo && value <= hi) ? value : forward.velocity_[i];
}

__global__ void Validate(const double *u, int n, int *bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(u[i]))
    atomicExch(bad, 1);
}

__global__ void DiffuseBoundary(Field f, double *next, double dt, double viscosity, double length, double speed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.nf)
    return;
  auto face = f.faces[i];
  if (face.aperture == 0) {
    next[i] = 0;
    return;
  }
  double h = sqrt(face.area), lap = 0;
  for (int direction = 0; direction < 3; ++direction)
    for (int sign = -1; sign <= 1; sign += 2) {
      auto p = face.centre;
      p[direction] += sign * h;
      for (int a = 0; a < 3; ++a)
        p[a] = Clamp(p[a], f.origin_[a] + (a == face.axis ? 0 : .5 * h),
                     f.origin_[a] + f.roots[a] * f.root_width_ - (a == face.axis ? 0 : .5 * h));
      double fraction = 0;
      if (direction != face.axis)
        for (int half = -1; half <= 1; half += 2) {
          auto wall = face.centre;
          wall[direction] += sign * .5 * h;
          wall[face.axis] += half * .5 * h;
          int other = f.NativeFace(wall, direction);
          if (other >= 0 && f.faces[other].aperture == 0)
            fraction += .5;
        }
      lap += (1 - fraction) * (f.VirtualFace(p, face.axis) - f.velocity_[i]);
    }
  double value = next[i] + dt * viscosity * lap / (h * h);
  if (length > 0) {
    double column = face.centre[0] - (face.axis == 0 ? 0 : .5 * h);
    double distance = f.origin_[0] + f.roots[0] * f.root_width_ - column;
    if (distance < length && column > f.origin_[0]) {
      double blend = Clamp(1 - distance / length, 0, 1);
      blend *= .25 * blend;
      value += blend * ((face.axis == 0 ? speed : 0) - value);
    }
  }
  if ((face.low < 0 || face.high < 0) && !(face.axis == 0 && face.high < 0))
    value = face.axis == 0 ? speed : 0;
  next[i] = value;
}

__global__ void ScalePressure(double *p, int n, double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] *= scale;
}

// Per-block deterministic sums avoid load atomics and transfer only 12 scalars.
__global__ void LoadPartials(Field f, const double *pressure, double density, double viscosity, double *partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double value[12]{};
  if (i < f.nc && f.cells[i].active) {
    Array<double, 3> normal{}, u{}, area{};
    for (int k = f.offsets[i]; k < f.offsets[i + 1]; ++k) {
      int id = f.incidence[k], j = abs(id) - 1;
      auto face = f.faces[j];
      int a = face.axis;
      double wetted = (id > 0 ? 1 : -1) * face.area * (1 - face.aperture), force = wetted * pressure[i];
      value[a] += force;
      value[3 + (a + 1) % 3] += face.centre[(a + 2) % 3] * force;
      value[3 + (a + 2) % 3] -= face.centre[(a + 1) % 3] * force;
      normal[a] += wetted;
      u[a] += face.area * f.velocity_[j];
      area[a] += face.area;
    }
    double wetted = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (wetted > 0) {
      double dot = 0, speed2 = 0;
      for (int a = 0; a < 3; ++a) {
        normal[a] /= wetted;
        u[a] /= area[a];
        dot += normal[a] * u[a];
      }
      for (int a = 0; a < 3; ++a) {
        u[a] -= dot * normal[a];
        speed2 += u[a] * u[a];
      }
      double wall = WallEffectiveViscosity(float(viscosity), float(sqrt(speed2)), float(.5 * f.cells[i].width));
      for (int a = 0; a < 3; ++a) {
        double force = u[a] * 2 * density * wall * wetted / f.cells[i].width;
        value[6 + a] += force;
        value[9 + (a + 1) % 3] += f.cells[i].centre[(a + 2) % 3] * force;
        value[9 + (a + 2) % 3] -= f.cells[i].centre[(a + 1) % 3] * force;
      }
    }
  }
  __shared__ double sum[256];
  for (int component = 0; component < 12; ++component) {
    sum[threadIdx.x] = value[component];
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
      if (threadIdx.x < stride)
        sum[threadIdx.x] += sum[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      partial[component * gridDim.x + blockIdx.x] = sum[0];
    __syncthreads();
  }
}

__global__ void ReduceLoads(const double *partial, int count, double *loads) {
  __shared__ double sum[256];
  double value = 0;
  for (int i = threadIdx.x; i < count; i += 256)
    value += partial[blockIdx.x * count + i];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    loads[blockIdx.x] = sum[0];
}

__global__ void SpeedPartials(Field f, const double *velocity, double *partial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double speed = 0;
  if (i < f.nc && f.cells[i].active) {
    Array<double, 3> u{}, area{};
    for (int k = f.offsets[i]; k < f.offsets[i + 1]; ++k) {
      int id = abs(f.incidence[k]) - 1;
      auto g = f.faces[id];
      u[g.axis] += g.area * velocity[id];
      area[g.axis] += g.area;
    }
    for (int a = 0; a < 3; ++a)
      u[a] /= area[a];
    speed = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (!isfinite(speed))
      speed = INFINITY;
  }
  __shared__ double values[256];
  values[threadIdx.x] = speed;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = Max(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x)
    partial[blockIdx.x] = values[0];
}

__global__ void SpeedMaximum(const double *partial, int n, double *output) {
  double value = 0;
  for (int i = threadIdx.x; i < n; i += 256)
    value = Max(value, partial[i]);
  __shared__ double values[256];
  values[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = Max(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x)
    *output = values[0];
}
}  // namespace

struct GpuTransport::Impl {
  int device{};
  Field fields[2]{};
  int *neighbors{}, *layers{}, *bad{};
  double *result{}, *pressure{}, *partial{}, *loads{};
  double min_area{INFINITY};
  std::vector<void *> allocations;
  std::string error;

  ~Impl() {
    cudaSetDevice(device);
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
  bool Allocate(T *&p, size_t n) {
    if (!Check(cudaMalloc(&p, n * sizeof(T))))
      return false;
    allocations.push_back(p);
    return true;
  }

  template <class T>
  bool Upload(const T *&p, const std::vector<T> &v) {
    T *q{};
    if (!Allocate(q, v.size()) || !Check(cudaMemcpy(q, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)))
      return false;
    p = q;
    return true;
  }

  bool Valid(const double *u, int count = 0) {
    if (!count)
      count = fields[0].nf;
    int host = 0;
    if (!Check(cudaMemset(bad, 0, sizeof(int))))
      return false;
    Validate<<<(count + 255) / 256, 256>>>(u, count, bad);
    if (!Check(cudaGetLastError()) || !Check(cudaMemcpy(&host, bad, sizeof(int), cudaMemcpyDeviceToHost)))
      return false;
    if (host) {
      error = "Non-finite transport field";
      return false;
    }
    return true;
  }

  bool Prepare(Field f) {
    int fb = (f.nf + 255) / 256, cb = (f.nc + 255) / 256;
    ClearSolid<<<fb, 256>>>(f);
    Extend<<<fb, 256>>>(f, neighbors, layers, 1);
    Extend<<<fb, 256>>>(f, neighbors, layers, 2);
    Centres<<<cb, 256>>>(f);
    Slopes<<<cb, 256>>>(f);
    return Check(cudaGetLastError());
  }
};

GpuTransport::GpuTransport() = default;
GpuTransport::~GpuTransport() = default;

std::string GpuTransport::Error() const {
  return !build_error_.empty() ? build_error_ : (impl_ ? impl_->error : "GPU transport is not built");
}

bool GpuTransport::Build(int device,
                         const Mesh &mesh,
                         const Topology &tree,
                         double width,
                         std::array<double, 3> origin) {
  auto next = std::make_unique<Impl>();
  next->device = device;
  auto fail = [&]() {
    build_error_ = next->error.empty() ? "Invalid transport geometry" : next->error;
    return false;
  };
  if (mesh.Cells().empty() || mesh.Cells().size() > size_t(INT_MAX) || mesh.Faces().size() > size_t(INT_MAX / 6) ||
      !(width > 0) || !std::isfinite(width) || tree.Leaves().size() > size_t(INT_MAX))
    return fail();
  for (double p : origin)
    if (!std::isfinite(p))
      return fail();
  auto &f = next->fields[0];
  f.n = mesh.CellsPerBlock();
  f.nc = int(mesh.Cells().size());
  f.nf = int(mesh.Faces().size());
  f.nb = int(tree.Leaves().size());
  f.max_level = tree.MaxLevel();
  f.root_width_ = width;
  if (size_t(f.n) * f.n * f.n * size_t(f.nb) != mesh.Cells().size())
    return fail();
  for (int a = 0; a < 3; ++a) {
    f.origin_[a] = origin[a];
    f.roots[a] = tree.Roots()[a];
  }
  std::vector<Cell> cells;
  std::vector<Face> faces;
  for (auto c : mesh.Cells())
    cells.push_back({{c.centre[0], c.centre[1], c.centre[2]}, c.width, c.active});
  for (auto g : mesh.Faces())
    next->min_area = std::min(next->min_area, g.area);
  for (auto g : mesh.Faces())
    faces.push_back({g.low, g.high, g.axis, {g.centre[0], g.centre[1], g.centre[2]}, g.area, g.aperture});
  if (!next->Check(cudaSetDevice(device)) || !next->Upload(f.cells, cells) || !next->Upload(f.faces, faces) ||
      !next->Upload(f.leaves, tree.Leaves()) || !next->Upload(f.offsets, mesh.Offsets()) ||
      !next->Upload(f.incidence, mesh.Incidence()))
    return fail();
  next->fields[1] = f;
  for (auto &field : next->fields)
    if (!next->Allocate(field.velocity_, f.nf) || !next->Allocate(field.centres_, f.nc) ||
        !next->Allocate(field.slopes_, f.nc) || !next->Allocate(field.limits_, f.nc))
      return fail();
  if (!next->Allocate(next->neighbors, size_t(f.nf) * 6) || !next->Allocate(next->layers, f.nf) ||
      !next->Allocate(next->bad, 1) || !next->Allocate(next->result, f.nf) || !next->Allocate(next->pressure, f.nc) ||
      !next->Allocate(next->partial, size_t((f.nc + 255) / 256) * 12) || !next->Allocate(next->loads, 12))
    return fail();
  SlipNeighbors<<<(f.nf + 255) / 256, 256>>>(f, next->neighbors, next->layers);
  if (!next->Check(cudaGetLastError()) || !next->Check(cudaDeviceSynchronize()))
    return fail();
  impl_ = std::move(next);
  build_error_.clear();
  return true;
}

bool GpuTransport::AdvectDevice(const double *velocity,
                                double *destination,
                                size_t faces,
                                double dt,
                                bool cubic,
                                bool corrected) {
  if (!impl_ || !velocity || !destination || faces != size_t(impl_->fields[0].nf) || !std::isfinite(dt))
    return false;
  auto &w = *impl_;
  w.error.clear();
  auto a = w.fields[0], b = w.fields[1];
  int blocks = (a.nf + 255) / 256;
  if (!w.Check(cudaSetDevice(w.device)) ||
      !w.Check(cudaMemcpy(a.velocity_, velocity, faces * sizeof(double), cudaMemcpyDeviceToDevice)) ||
      !w.Valid(a.velocity_) || !w.Prepare(a))
    return false;
  Forward<<<blocks, 256>>>(a, dt, cubic, b.velocity_);
  if (!w.Check(cudaGetLastError()) || !w.Valid(b.velocity_))
    return false;
  const double *result = b.velocity_;
  if (corrected) {
    if (!w.Prepare(b))
      return false;
    Correct<<<blocks, 256>>>(a, b, dt, cubic, w.result);
    result = w.result;
  }
  return w.Check(cudaGetLastError()) && w.Valid(result) &&
         (destination == result ||
          w.Check(cudaMemcpy(destination, result, faces * sizeof(double), cudaMemcpyDeviceToDevice)));
}

AdvanceResult GpuTransport::AdvanceDevice(GpuProjection &projection,
                                          double *velocity,
                                          size_t faces,
                                          double *pressure,
                                          size_t cells,
                                          double dt,
                                          double speed,
                                          bool cubic,
                                          double density,
                                          double wall_viscosity,
                                          double viscosity,
                                          double sponge_length) {
  AdvanceResult result;
  if (!impl_ || !velocity || !pressure || velocity == pressure || faces != size_t(impl_->fields[0].nf) ||
      cells != size_t(impl_->fields[0].nc) || !(dt > 0) || !std::isfinite(dt) || !std::isfinite(speed) ||
      !(density > 0) || !std::isfinite(density) || !(wall_viscosity > 0) || !std::isfinite(wall_viscosity) ||
      viscosity < 0 || !std::isfinite(viscosity) || sponge_length < 0 || !std::isfinite(sponge_length) ||
      dt * viscosity / impl_->min_area > 1.0 / 6)
    return result;
  auto &w = *impl_;
  if (!AdvectDevice(velocity, w.result, faces, dt, cubic, true))
    return result;
  auto f = w.fields[0];
  int fb = (f.nf + 255) / 256, cb = (f.nc + 255) / 256;
  DiffuseBoundary<<<fb, 256>>>(f, w.result, dt, viscosity, sponge_length, speed);
  if (!w.Check(cudaGetLastError()) || !w.Valid(w.result))
    return result;
  result.projection = projection.ProjectDevice(w.result, faces, w.pressure, cells, 1e-8);
  if (!result.projection.converged) {
    w.error = projection.Error();
    return result;
  }
  ScalePressure<<<cb, 256>>>(w.pressure, f.nc, density / dt);
  if (!w.Check(cudaGetLastError()) || !w.Valid(w.pressure, f.nc))
    return result;
  LoadPartials<<<cb, 256>>>(f, w.pressure, density, wall_viscosity, w.partial);
  ReduceLoads<<<12, 256>>>(w.partial, cb, w.loads);
  double loads[12];
  if (!w.Check(cudaGetLastError()) || !w.Check(cudaMemcpy(loads, w.loads, sizeof(loads), cudaMemcpyDeviceToHost)))
    return result;
  for (double v : loads)
    if (!std::isfinite(v)) {
      w.error = "Non-finite GPU loads";
      return result;
    }
  for (int k = 0; k < 6; ++k) {
    result.pressure_loads[k] = loads[k];
    result.friction_loads[k] = loads[6 + k];
    result.total_loads[k] = loads[k] + loads[6 + k];
  }
  if (!w.Check(cudaMemcpy(velocity, w.result, faces * sizeof(double), cudaMemcpyDeviceToDevice)) ||
      !w.Check(cudaMemcpy(pressure, w.pressure, cells * sizeof(double), cudaMemcpyDeviceToDevice)))
    return result;
  result.advanced = true;
  return result;
}

bool GpuTransport::MaximumSpeedDevice(const double *velocity, size_t faces, double &speed) {
  if (!impl_ || !velocity || faces != size_t(impl_->fields[0].nf))
    return false;
  auto &w = *impl_;
  if (!w.Check(cudaSetDevice(w.device)))
    return false;
  auto f = w.fields[0];
  int blocks = (f.nc + 255) / 256;
  SpeedPartials<<<blocks, 256>>>(f, velocity, w.partial);
  SpeedMaximum<<<1, 256>>>(w.partial, blocks, w.loads);
  double value;
  if (!w.Check(cudaGetLastError()) || !w.Check(cudaMemcpy(&value, w.loads, sizeof(double), cudaMemcpyDeviceToHost)))
    return false;
  if (!std::isfinite(value)) {
    w.error = "Non-finite cell speed";
    return false;
  }
  speed = value;
  return true;
}
}  // namespace wind3d::amr
