#include "level_set_grid.h"

void MeshData::SaveToFile(std::string filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  for (size_t i = 0; i < vertex_data.size() / 3; i++) {
    file << "v " << vertex_data[3 * i + 0] << " " << vertex_data[3 * i + 1] << " " << vertex_data[3 * i + 2] << "\n";
  }

  for (size_t i = 0; i < index_data.size() / 3; i++) {
    file << "f " << index_data[3 * i + 0] + 1 << " " << index_data[3 * i + 1] + 1 << " " << index_data[3 * i + 2] + 1
         << "\n";
  }

  file.close();
}

LevelSetBase::LevelSetBase(float *data,
                           int size_x,
                           int size_y,
                           int size_z,
                           float delta_x,
                           float offset_x,
                           float offset_y,
                           float offset_z)
    : data_(data),
      size_x_(size_x),
      size_y_(size_y),
      size_z_(size_z),
      delta_x_(delta_x),
      offset_x_(offset_x),
      offset_y_(offset_y),
      offset_z_(offset_z) {
}

__device__ __host__ int LevelSetBase::Index(int x, int y, int z) const {
  return x + size_x_ * (y + size_y_ * z);
}

__device__ float &LevelSetBase::operator()(int x, int y, int z) {
  return data_[x + size_x_ * (y + size_y_ * z)];
}

__device__ const float &LevelSetBase::operator()(int x, int y, int z) const {
  return data_[x + size_x_ * (y + size_y_ * z)];
}

__device__ float &LevelSetBase::ClampAt(int x, int y, int z) {
  x = std::max(int(0), std::min(x, int(size_x_ - 1)));
  y = std::max(int(0), std::min(y, int(size_y_ - 1)));
  z = std::max(int(0), std::min(z, int(size_z_ - 1)));
  return data_[x + size_x_ * (y + size_y_ * z)];
}

__device__ const float &LevelSetBase::ClampAt(int x, int y, int z) const {
  x = std::max(int(0), std::min(x, int(size_x_ - 1)));
  y = std::max(int(0), std::min(y, int(size_y_ - 1)));
  z = std::max(int(0), std::min(z, int(size_z_ - 1)));
  return data_[x + size_x_ * (y + size_y_ * z)];
}

__device__ float LevelSetBase::SampleAt(float x, float y, float z) const {
  int ix = floor((x - offset_x_) / delta_x_);
  int iy = floor((y - offset_y_) / delta_x_);
  int iz = floor((z - offset_z_) / delta_x_);
  float fx = x - ix;
  float fy = y - iy;
  float fz = z - iz;
  return (1 - fx) * (1 - fy) * (1 - fz) * ClampAt(ix, iy, iz) + fx * (1 - fy) * (1 - fz) * ClampAt(ix + 1, iy, iz) +
         (1 - fx) * fy * (1 - fz) * ClampAt(ix, iy + 1, iz) + fx * fy * (1 - fz) * ClampAt(ix + 1, iy + 1, iz) +
         (1 - fx) * (1 - fy) * fz * ClampAt(ix, iy, iz + 1) + fx * (1 - fy) * fz * ClampAt(ix + 1, iy, iz + 1) +
         (1 - fx) * fy * fz * ClampAt(ix, iy + 1, iz + 1) + fx * fy * fz * ClampAt(ix + 1, iy + 1, iz + 1);
}

LevelSetGrid::LevelSetGrid(int size_x,
                           int size_y,
                           int size_z,
                           float delta_x,
                           float offset_x,
                           float offset_y,
                           float offset_z)
    : LevelSetBase(nullptr, size_x, size_y, size_z, delta_x, offset_x, offset_y, offset_z) {
  printf("LevelSetGrid: size=(%zu, %zu, %zu), delta_x=%f, offset=(%f, %f, %f)\n", size_x_, size_y_, size_z_, delta_x_,
         offset_x_, offset_y_, offset_z_);
  if (size_x && size_y && size_z) {
    int size = size_x_ * size_y_ * size_z_;
    cudaMalloc(&data_, size * sizeof(float));
  }
}

LevelSetGrid::LevelSetGrid(const LevelSetGrid &other)
    : LevelSetGrid(other.size_x_,
                   other.size_y_,
                   other.size_z_,
                   other.delta_x_,
                   other.offset_x_,
                   other.offset_y_,
                   other.offset_z_) {
  if (data_ && other.data_) {
    int size = size_x_ * size_y_ * size_z_;
    cudaMemcpy(data_, other.data_, size * sizeof(float), cudaMemcpyDeviceToDevice);
  }
}

LevelSetGrid::LevelSetGrid(LevelSetGrid &&other) noexcept
    : LevelSetBase(other.data_,
                   other.size_x_,
                   other.size_y_,
                   other.size_z_,
                   other.delta_x_,
                   other.offset_x_,
                   other.offset_y_,
                   other.offset_z_) {
  other.data_ = nullptr;
  other.size_x_ = 0;
  other.size_y_ = 0;
  other.size_z_ = 0;
  other.delta_x_ = 0.0f;
  other.offset_x_ = 0.0f;
  other.offset_y_ = 0.0f;
}

LevelSetGrid &LevelSetGrid::operator=(const LevelSetGrid &other) {
  size_x_ = other.size_x_;
  size_y_ = other.size_y_;
  size_z_ = other.size_z_;
  delta_x_ = other.delta_x_;
  offset_x_ = other.offset_x_;
  offset_y_ = other.offset_y_;
  offset_z_ = other.offset_z_;
  if (data_) {
    cudaFree(data_);
    data_ = nullptr;
  }
  if (size_x_ && size_y_ && size_z_) {
    int size = size_x_ * size_y_ * size_z_;
    cudaMalloc(&data_, size * sizeof(float));
    if (other.data_) {
      cudaMemcpy(data_, other.data_, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
  }
  return *this;
}

LevelSetGrid &LevelSetGrid::operator=(LevelSetGrid &&other) noexcept {
  size_x_ = other.size_x_;
  size_y_ = other.size_y_;
  size_z_ = other.size_z_;
  delta_x_ = other.delta_x_;
  offset_x_ = other.offset_x_;
  offset_y_ = other.offset_y_;
  offset_z_ = other.offset_z_;
  if (data_) {
    cudaFree(data_);
  }
  if (other.data_) {
    data_ = other.data_;
    other.data_ = nullptr;
  } else {
    data_ = nullptr;
  }
  return *this;
}

LevelSetGrid::~LevelSetGrid() {
  if (data_) {
    cudaFree(data_);
  }
}

std::vector<float> LevelSetGrid::Download() const {
  std::vector<float> host_data;
  if (data_) {
    int size = size_x_ * size_y_ * size_z_;
    host_data.resize(size);
    cudaMemcpy(host_data.data(), data_, size * sizeof(float), cudaMemcpyDeviceToHost);
  }
  return host_data;
}

struct EdgeIndex {
  int ex;
  int ey;
  int ez;
};

__global__ void GenerateVerticesKernel(const LevelSetBase base,
                                       float *vertex_buffer,
                                       EdgeIndex *edge_indices,
                                       int *vertex_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;
  if (idx < base.SizeX() || idy < base.SizeY() || idz < base.SizeZ()) {
    EdgeIndex edge_index{-1, -1, -1};
    float value = base(idx, idy, idz);
    float value1 = value;
    float ratio;
    if (idx + 1 < base.SizeX()) {
      value1 = base(idx + 1, idy, idz);
      if ((value >= 0.0f) ^ (value1 >= 0.0f)) {
        int vtx_id = atomicAdd(vertex_count, 1);
        if (vertex_buffer) {
          ratio = value / (value - value1);
          vertex_buffer[3 * vtx_id + 0] = base.OffsetX() + (idx + ratio) * base.DeltaX();
          vertex_buffer[3 * vtx_id + 1] = base.OffsetY() + idy * base.DeltaX();
          vertex_buffer[3 * vtx_id + 2] = base.OffsetZ() + idz * base.DeltaX();
          edge_index.ex = vtx_id;
        }
      }
    }

    if (idy + 1 < base.SizeY()) {
      value1 = base(idx, idy + 1, idz);
      if ((value >= 0.0f) ^ (value1 >= 0.0f)) {
        int vtx_id = atomicAdd(vertex_count, 1);
        if (vertex_buffer) {
          ratio = value / (value - value1);
          vertex_buffer[3 * vtx_id + 0] = base.OffsetX() + idx * base.DeltaX();
          vertex_buffer[3 * vtx_id + 1] = base.OffsetY() + (idy + ratio) * base.DeltaX();
          vertex_buffer[3 * vtx_id + 2] = base.OffsetZ() + idz * base.DeltaX();
          edge_index.ey = vtx_id;
        }
      }
    }

    if (idz + 1 < base.SizeZ()) {
      value1 = base(idx, idy, idz + 1);
      if ((value >= 0.0f) ^ (value1 >= 0.0f)) {
        int vtx_id = atomicAdd(vertex_count, 1);
        if (vertex_buffer) {
          ratio = value / (value - value1);
          vertex_buffer[3 * vtx_id + 0] = base.OffsetX() + idx * base.DeltaX();
          vertex_buffer[3 * vtx_id + 1] = base.OffsetY() + idy * base.DeltaX();
          vertex_buffer[3 * vtx_id + 2] = base.OffsetZ() + (idz + ratio) * base.DeltaX();
          edge_index.ez = vtx_id;
        }
      }
    }

    if (edge_indices) {
      edge_indices[base.Index(idx, idy, idz)] = edge_index;
    }
  }
}

namespace {

// table referenced from: https://paulbourke.net/geometry/polygonise/

__constant__ int mc_tri_table[256][16];
int mc_triangle_table[256][16] = {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
                                  {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
                                  {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
                                  {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
                                  {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
                                  {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
                                  {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
                                  {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
                                  {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
                                  {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
                                  {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
                                  {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
                                  {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
                                  {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
                                  {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
                                  {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
                                  {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
                                  {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
                                  {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
                                  {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
                                  {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
                                  {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
                                  {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
                                  {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
                                  {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
                                  {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
                                  {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
                                  {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
                                  {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
                                  {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
                                  {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
                                  {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
                                  {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
                                  {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
                                  {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
                                  {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
                                  {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
                                  {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
                                  {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
                                  {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
                                  {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
                                  {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
                                  {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
                                  {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
                                  {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
                                  {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
                                  {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
                                  {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
                                  {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
                                  {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
                                  {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
                                  {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
                                  {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
                                  {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
                                  {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
                                  {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
                                  {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
                                  {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
                                  {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
                                  {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
                                  {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
                                  {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
                                  {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
                                  {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
                                  {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
                                  {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
                                  {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
                                  {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
                                  {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
                                  {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
                                  {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
                                  {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
                                  {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
                                  {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
                                  {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
                                  {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
                                  {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
                                  {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
                                  {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
                                  {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
                                  {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
                                  {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
                                  {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
                                  {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
                                  {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
                                  {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
                                  {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
                                  {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
                                  {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
                                  {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
                                  {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
                                  {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
                                  {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
                                  {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
                                  {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
                                  {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
                                  {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
                                  {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
                                  {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
                                  {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
                                  {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
                                  {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
                                  {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
                                  {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
                                  {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
                                  {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
                                  {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
                                  {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
                                  {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
                                  {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
                                  {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
}  // namespace

__global__ void GenerateIndicesKernel(const LevelSetBase grid,
                                      EdgeIndex *edge_indices,
                                      uint32_t *index_buffer,
                                      int *index_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  float value[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  int edge_index[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  if (idx < grid.SizeX() && idy < grid.SizeY() && idz < grid.SizeZ()) {
    value[0] = grid(idx, idy, idz);
    auto eis = edge_indices[grid.Index(idx, idy, idz)];
    edge_index[0] = eis.ex;
    edge_index[3] = eis.ey;
    edge_index[8] = eis.ez;
  }
  if (idx + 1 < grid.SizeX() && idy < grid.SizeY() && idz < grid.SizeZ()) {
    value[1] = grid(idx + 1, idy, idz);
    auto eis = edge_indices[grid.Index(idx + 1, idy, idz)];
    edge_index[1] = eis.ey;
    edge_index[9] = eis.ez;
  }
  if (idx + 1 < grid.SizeX() && idy + 1 < grid.SizeY() && idz < grid.SizeZ()) {
    value[2] = grid(idx + 1, idy + 1, idz);
    auto eis = edge_indices[grid.Index(idx + 1, idy + 1, idz)];
    edge_index[10] = eis.ez;
  }
  if (idx < grid.SizeX() && idy + 1 < grid.SizeY() && idz < grid.SizeZ()) {
    value[3] = grid(idx, idy + 1, idz);
    auto eis = edge_indices[grid.Index(idx, idy + 1, idz)];
    edge_index[2] = eis.ex;
    edge_index[11] = eis.ez;
  }
  if (idx < grid.SizeX() && idy < grid.SizeY() && idz + 1 < grid.SizeZ()) {
    value[4] = grid(idx, idy, idz + 1);
    auto eis = edge_indices[grid.Index(idx, idy, idz + 1)];
    edge_index[4] = eis.ex;
    edge_index[7] = eis.ey;
  }
  if (idx + 1 < grid.SizeX() && idy < grid.SizeY() && idz + 1 < grid.SizeZ()) {
    value[5] = grid(idx + 1, idy, idz + 1);
    auto eis = edge_indices[grid.Index(idx + 1, idy, idz + 1)];
    edge_index[5] = eis.ey;
  }
  if (idx + 1 < grid.SizeX() && idy + 1 < grid.SizeY() && idz + 1 < grid.SizeZ()) {
    value[6] = grid(idx + 1, idy + 1, idz + 1);
  }
  if (idx < grid.SizeX() && idy + 1 < grid.SizeY() && idz + 1 < grid.SizeZ()) {
    value[7] = grid(idx, idy + 1, idz + 1);
    auto eis = edge_indices[grid.Index(idx, idy + 1, idz + 1)];
    edge_index[6] = eis.ex;
  }
  int cube_index = 0;
  for (int i = 0; i < 8; ++i) {
    if (value[i] >= 0.0f) {
      cube_index |= (1 << i);
    }
  }

  for (int i = 0; mc_tri_table[cube_index][i] != -1; i += 3) {
    int e0 = mc_tri_table[cube_index][i];
    int e1 = mc_tri_table[cube_index][i + 1];
    int e2 = mc_tri_table[cube_index][i + 2];
    if (edge_index[e0] >= 0 && edge_index[e1] >= 0 && edge_index[e2] >= 0) {
      int index = atomicAdd(index_count, 3);
      if (index_buffer) {
        index_buffer[index] = edge_index[e0];
        index_buffer[index + 1] = edge_index[e1];
        index_buffer[index + 2] = edge_index[e2];
      }
    }
  }
}

MeshData LevelSetGrid::MarchingCubes() const {
  MeshData result;
  int *d_vertex_count = nullptr;
  float *d_vertex_data = nullptr;
  int *d_index_count = nullptr;
  uint32_t *d_index_data = nullptr;
  EdgeIndex *d_edge_indices = nullptr;
  DeferredProcess defer([&]() {
    if (d_vertex_count) {
      cudaFree(d_vertex_count);
    }
    if (d_vertex_data) {
      cudaFree(d_vertex_data);
    }
    if (d_index_count) {
      cudaFree(d_index_count);
    }
    if (d_index_data) {
      cudaFree(d_index_data);
    }
    if (d_edge_indices) {
      cudaFree(d_edge_indices);
    }
  });
  cudaMalloc(&d_vertex_count, sizeof(int));
  cudaMemset(d_vertex_count, 0, sizeof(int));
  dim3 block(8, 8, 8);
  dim3 grid((size_x_ + block.x - 1) / block.x, (size_y_ + block.y - 1) / block.y, (size_z_ + block.z - 1) / block.z);
  GenerateVerticesKernel<<<grid, block>>>(*this, nullptr, nullptr, d_vertex_count);
  int h_vertex_count = 0;
  cudaMemcpy(&h_vertex_count, d_vertex_count, sizeof(int), cudaMemcpyDeviceToHost);
  printf("MarchingCubes: vertex count = %d\n", h_vertex_count);
  cudaMalloc(&d_vertex_data, h_vertex_count * 3 * sizeof(float));
  cudaMemset(d_vertex_count, 0, sizeof(int));
  cudaMalloc(&d_edge_indices, size_x_ * size_y_ * size_z_ * sizeof(EdgeIndex));
  GenerateVerticesKernel<<<grid, block>>>(*this, d_vertex_data, d_edge_indices, d_vertex_count);
  result.vertex_data.resize(h_vertex_count * 3);
  cudaMemcpy(result.vertex_data.data(), d_vertex_data, h_vertex_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpyToSymbol(mc_tri_table, mc_triangle_table, 256 * 16 * sizeof(int));

  cudaMalloc(&d_index_count, sizeof(int));
  cudaMemset(d_index_count, 0, sizeof(int));

  GenerateIndicesKernel<<<grid, block>>>(*this, d_edge_indices, nullptr, d_index_count);
  int h_index_count = 0;
  cudaMemcpy(&h_index_count, d_index_count, sizeof(int), cudaMemcpyDeviceToHost);
  printf("MarchingCubes: index count = %d\n", h_index_count);
  cudaMalloc(&d_index_data, h_index_count * sizeof(uint32_t));
  cudaMemset(d_index_count, 0, sizeof(int));
  GenerateIndicesKernel<<<grid, block>>>(*this, d_edge_indices, d_index_data, d_index_count);
  result.index_data.resize(h_index_count);
  cudaMemcpy(result.index_data.data(), d_index_data, h_index_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  return result;
}
