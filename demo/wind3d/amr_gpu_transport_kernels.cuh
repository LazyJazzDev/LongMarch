#pragma once
// Internal CUDA sampling and advection kernels shared by transport backends.
// Each translation unit owns its kernel symbols; this is not a public API.
#include <cuda_runtime.h>

#include <cuda/std/array>

#include "amr_mesh.h"
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

struct HaloSite {
  int cell, native;
};

// Global IDs resolve to owned storage. Null maps retain the contiguous
// single-device path. Peer reads are valid only after the owning stage ends.
template <class T>
struct DistributedValues {
  T *local{};
  T *const *peers{};
  const int *owners{}, *indices{};

  __device__ T &operator[](int id) const {
    return peers ? peers[owners[id]][indices[id]] : local[id];
  }
};

struct Field {
  int n, max_level, nb, nc, nf;
  Array<int, 3> roots;
  Array<double, 3> origin_;
  double root_width_;
  const Block *leaves;
  const Cell *cells;
  const Face *faces;
  const int *offsets, *incidence;
  DistributedValues<double> velocity_;
  DistributedValues<Array<double, 3>> centres_;
  DistributedValues<Array<double, 9>> slopes_;
  DistributedValues<Array<double, 6>> limits_;
  const int *owned_faces{}, *owned_cells{};
  int face_count{}, cell_count{};

  __device__ int FaceWork(int i) const {
    return owned_faces ? (i < face_count ? owned_faces[i] : -1) : (i < nf ? i : -1);
  }

  __device__ int CellWork(int i) const {
    return owned_cells ? (i < cell_count ? owned_cells[i] : -1) : (i < nc ? i : -1);
  }

  HaloSite *sites{};
  double *halo{};
  int halo_count{};
  bool prescribed_inlet{};
  double inlet_speed{};

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

  __device__ double VirtualInlet(Array<double, 3> point, int axis) const {
    bool reflected = prescribed_inlet && point[0] < origin_[0];
    if (reflected)
      point[0] = 2 * origin_[0] - point[0];
    double value = VirtualFace(point, axis);
    return reflected ? 2 * (axis == 0 ? inlet_speed : 0) - value : value;
  }

  __device__ double DirectComponent(Array<double, 3> point, int axis, bool cubic, double spacing = 0) const {
    for (int a = 0; a < 3; ++a) {
      if (!isfinite(point[a]))
        return nan("");
      point[a] = Clamp(point[a], origin_[a], origin_[a] + roots[a] * root_width_);
    }

    int cell = CellAt(point);
    if (cell < 0)
      return 0;
    double h = spacing > 0 ? spacing : cells[cell].width;
    double base[3], fraction[3];
    for (int a = 0; a < 3; ++a) {
      double q = (point[a] - origin_[a]) / h - (a == axis ? 0 : .5);
      base[a] = floor(q);
      fraction[a] = q - base[a];
    }
    auto corner_point = [&](int x, int y, int z) {
      int corner[3]{x, y, z};
      Array<double, 3> p{};
      for (int a = 0; a < 3; ++a) {
        p[a] = fma(base[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]);
        if (a != 0 || !prescribed_inlet || p[a] >= origin_[0])
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
              values[x] = VirtualInlet(corner_point(x - 1, y - 1, z - 1), axis);
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
            p[a] = fma(base[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]);
            weight *= corner[a] ? fraction[a] : 1 - fraction[a];
            double lower = origin_[a] + (a == axis ? 0 : .5 * h),
                   upper = origin_[a] + roots[a] * root_width_ - (a == axis ? 0 : .5 * h);
            if (a != 0 || !prescribed_inlet || p[a] >= origin_[0])
              p[a] = Clamp(p[a], lower, upper);
          }
          double value = VirtualInlet(p, axis);
          result += weight * value;
          int f = NativeFace(p, axis);
          double aperture = f < 0 ? 1 : faces[f].aperture;
          weighted += aperture * weight * value;
          total += aperture * weight;
        }
    return total > 1e-3 ? weighted / total : result;
  }

  __device__ Array<double, 3> HaloPoint(int index, int &axis) const {
    int side = n + 5, cube = side * side * side, block = index / (3 * cube), slot = index % cube;
    axis = (index / cube) % 3;
    auto b = leaves[block];
    int lower[3]{b.x * n, b.y * n, b.z * n};
    int corner[3]{slot % side - 2, slot / side % side - 2, slot / (side * side) - 2};
    double h = cells[block * n * n * n].width;
    Array<double, 3> p;
    for (int a = 0; a < 3; ++a)
      p[a] =
          Clamp(fma(lower[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]), origin_[a] + (a == axis ? 0 : .5 * h),
                origin_[a] + roots[a] * root_width_ - (a == axis ? 0 : .5 * h));
    double raw_x = fma(lower[0] + corner[0] + (axis == 0 ? 0 : .5), h, origin_[0]);
    if (prescribed_inlet && raw_x < origin_[0])
      p[0] = raw_x;
    return p;
  }

  __device__ int StencilIndex(int cell, const double *base, int x, int y, int z, int axis) const {
    int side = n + 5, block = cell / (n * n * n);
    auto b = leaves[block];
    int i = int(base[0]) - b.x * n + x + 2, j = int(base[1]) - b.y * n + y + 2, k = int(base[2]) - b.z * n + z + 2;
    if (i < 0 || i >= side || j < 0 || j >= side || k < 0 || k >= side)
      return -1;
    return (block * 3 + axis) * side * side * side + (k * side + j) * side + i;
  }

  __device__ double StencilValue(int cell, const double *base, int x, int y, int z, int axis, double spacing = 0)
      const {
    double h = spacing > 0 ? spacing : cells[cell].width;
    bool ghost = prescribed_inlet && fma(base[0] + x + (axis == 0 ? 0 : .5), h, origin_[0]) < origin_[0];
    if (halo && h == cells[cell].width) {
      int index = StencilIndex(cell, base, x, y, z, axis);
      if (index >= 0)
        return halo[index];
    }
    int corner[3]{x, y, z};
    Array<double, 3> p;
    for (int a = 0; a < 3; ++a) {
      p[a] = fma(base[a] + corner[a] + (a == axis ? 0 : .5), h, origin_[a]);
      if (a != 0 || !ghost)
        p[a] = Clamp(p[a], origin_[a] + (a == axis ? 0 : .5 * h),
                     origin_[a] + roots[a] * root_width_ - (a == axis ? 0 : .5 * h));
    }
    return VirtualInlet(p, axis);
  }

  __device__ int FaceCell(Face face) const {
    if (face.low < 0)
      return face.high;
    if (face.high < 0)
      return face.low;
    return cells[face.low].width <= cells[face.high].width ? face.low : face.high;
  }

  __device__ double Component(Array<double, 3> point, int axis, bool cubic, double spacing = 0, int owner = -1) const {
    if (!halo)
      return DirectComponent(point, axis, cubic, spacing);
    for (int a = 0; a < 3; ++a) {
      if (!isfinite(point[a]))
        return nan("");
      point[a] = Clamp(point[a], origin_[a], origin_[a] + roots[a] * root_width_);
    }
    // The target block halo also covers short traces across its level boundary.
    int cell = owner >= 0 ? owner : CellAt(point);
    if (cell < 0)
      return 0;
    if (spacing > 0 && spacing != cells[cell].width)
      return DirectComponent(point, axis, cubic, spacing);
    double h = cells[cell].width, base[3], fraction[3];

    for (int a = 0; a < 3; ++a) {
      double q = (point[a] - origin_[a]) / h - (a == axis ? 0 : .5);
      base[a] = floor(q);
      fraction[a] = q - base[a];
    }
    // Defensive fallback for coordinates that round outside the owning block's halo.
    if (StencilIndex(cell, base, -1, -1, -1, axis) < 0 || StencilIndex(cell, base, 2, 2, 2, axis) < 0)
      return DirectComponent(point, axis, cubic, spacing);
    int side = n + 5, origin_index = StencilIndex(cell, base, 0, 0, 0, axis);
    if (prescribed_inlet && base[0] + (axis == 0 ? 0 : .5) < 0)
      cubic = false;
    if (cubic) {
      for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
          for (int x = 0; x < 2; ++x) {
            int native = sites[origin_index + (z * side + y) * side + x].native;
            if (native < 0 || faces[native].aperture != 1)
              cubic = false;
          }
      if (cubic) {
        double planes[4];
        for (int z = 0; z < 4; ++z) {
          double rows[4];
          for (int y = 0; y < 4; ++y) {
            double values[4];
            for (int x = 0; x < 4; ++x)
              values[x] = halo[origin_index + ((z - 1) * side + y - 1) * side + x - 1];
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
          int corner[3]{x, y, z}, index = origin_index + (z * side + y) * side + x, native = sites[index].native;
          if (prescribed_inlet && base[0] + x + (axis == 0 ? 0 : .5) < 0)
            native = -1;
          double weight = 1;
          for (int a = 0; a < 3; ++a)
            weight *= corner[a] ? fraction[a] : 1 - fraction[a];
          double value = halo[index], aperture = native < 0 ? 1 : faces[native].aperture;
          result += weight * value;
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

__global__ void MapHalos(Field f) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= f.halo_count)
    return;
  int axis;
  auto p = f.HaloPoint(index, axis);
  f.sites[index] = {f.CellAt(p), f.NativeFace(p, axis)};
}

__global__ void FillHalos(Field f) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= f.halo_count)
    return;
  int ghost_axis;
  auto ghost_point = f.HaloPoint(index, ghost_axis);
  if (f.prescribed_inlet && ghost_point[0] < f.origin_[0]) {
    f.halo[index] = f.VirtualInlet(ghost_point, ghost_axis);
    return;
  }
  auto site = f.sites[index];
  double value = 0;
  if (site.native >= 0)
    value = f.velocity_[site.native];
  else if (site.cell >= 0) {
    int axis;
    auto p = f.HaloPoint(index, axis);
    auto c = f.cells[site.cell];
    value = f.centres_[site.cell][axis];
    for (int a = 0; a < 3; ++a)
      value += f.slopes_[site.cell][3 * axis + a] * (p[a] - c.centre[a]);
    value = Clamp(value, f.limits_[site.cell][2 * axis], f.limits_[site.cell][2 * axis + 1]);
  }
  f.halo[index] = value;
}

__global__ void Centres(Field f) {
  int cell = f.CellWork(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell < 0)
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
  int cell = f.CellWork(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell < 0)
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
  int i = f.FaceWork(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= 0 && f.faces[i].aperture == 0)
    f.velocity_[i] = 0;
}

__global__ void Extend(Field f, const int *neighbors, const int *layers, int pass) {
  int i = f.FaceWork(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < 0 || f.faces[i].aperture > 0 || (pass == 1 ? layers[i] != 1 : layers[i] == 1))
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
  int local = blockIdx.x * blockDim.x + threadIdx.x;
  int i = f.FaceWork(local);
  if (i < 0)
    return;
  auto face = f.faces[i];
  out[local] = face.aperture == 0
                   ? 0
                   : f.Component(f.Departure(face.centre, dt), face.axis, cubic, sqrt(face.area), f.FaceCell(face));
}

__global__ void Correct(Field original, Field forward, double dt, bool cubic, double *out) {
  int local = blockIdx.x * blockDim.x + threadIdx.x;
  int i = original.FaceWork(local);
  if (i < 0)
    return;
  auto face = original.faces[i];
  if (face.aperture == 0) {
    out[local] = 0;
    return;
  }
  double value = forward.velocity_[i] +
                 .5 * (original.velocity_[i] - forward.Component(original.Departure(face.centre, -dt), face.axis, cubic,
                                                                 sqrt(face.area), forward.FaceCell(face)));
  if (!isfinite(value)) {
    out[local] = nan("");
    return;
  }
  auto back = original.Departure(face.centre, dt);
  for (int a = 0; a < 3; ++a)
    back[a] = Clamp(back[a], original.origin_[a], original.origin_[a] + original.roots[a] * original.root_width_);
  int cell = original.FaceCell(face);
  double h = sqrt(face.area), base[3];
  for (int a = 0; a < 3; ++a)
    base[a] = floor((back[a] - original.origin_[a]) / h - (a == face.axis ? 0 : .5));
  double lo = INFINITY, hi = -INFINITY;
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        double v = original.StencilValue(cell, base, x, y, z, face.axis, h);
        lo = Min(lo, v);
        hi = Max(hi, v);
      }
  out[local] = (value >= lo && value <= hi) ? value : forward.velocity_[i];
}

__global__ void DiffuseBoundary(Field f, double *next, double dt, double viscosity, double length, double speed) {
  int local = blockIdx.x * blockDim.x + threadIdx.x;
  int i = f.FaceWork(local);
  if (i < 0)
    return;
  auto face = f.faces[i];
  if (face.aperture == 0) {
    next[local] = 0;
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
  double value = next[local] + dt * viscosity * lap / (h * h);
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
  next[local] = value;
}

// Per-block deterministic sums avoid load atomics and transfer only 12 scalars.
__global__ void LoadPartials(Field f, const double *pressure, double density, double viscosity, double *partial) {
  int local = blockIdx.x * blockDim.x + threadIdx.x;
  int i = f.CellWork(local);
  double value[12]{};
  if (i >= 0 && f.cells[i].active) {
    Array<double, 3> normal{}, u{}, area{};
    for (int k = f.offsets[i]; k < f.offsets[i + 1]; ++k) {
      int id = f.incidence[k], j = abs(id) - 1;
      auto face = f.faces[j];
      int a = face.axis;
      double wetted = (id > 0 ? 1 : -1) * face.area * (1 - face.aperture), force = wetted * pressure[local];
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

}  // namespace
}  // namespace wind3d::amr
