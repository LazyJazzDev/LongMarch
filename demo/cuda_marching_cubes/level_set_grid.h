#include "cuda_runtime.h"
#include "long_march.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/reduce.h"
#include "thrust/sort.h"
#include "thrust/transform.h"

using namespace long_march;

struct MeshData {
  std::vector<float> vertex_data;
  std::vector<uint32_t> index_data;

  void SaveToFile(std::string filename);
};

class LevelSetBase {
 public:
  LevelSetBase(float *data = nullptr,
               int size_x = 0,
               int size_y = 0,
               int size_z = 0,
               float delta_x = 0.0f,
               float offset_x = 0.0f,
               float offset_y = 0.0f,
               float offset_z = 0.0f);
  __device__ __host__ int Index(int x, int y, int z) const;
  __device__ float &operator()(int x, int y, int z);
  __device__ const float &operator()(int x, int y, int z) const;
  __device__ float &ClampAt(int x, int y, int z);
  __device__ const float &ClampAt(int x, int y, int z) const;
  __device__ float SampleAt(float x, float y, float z) const;
  __device__ __host__ int SizeX() const {
    return size_x_;
  }
  __device__ __host__ int SizeY() const {
    return size_y_;
  }
  __device__ __host__ int SizeZ() const {
    return size_z_;
  }
  __device__ __host__ float DeltaX() const {
    return delta_x_;
  }
  __device__ __host__ float OffsetX() const {
    return offset_x_;
  }
  __device__ __host__ float OffsetY() const {
    return offset_y_;
  }
  __device__ __host__ float OffsetZ() const {
    return offset_z_;
  }

 protected:
  float *data_;
  int size_x_;
  int size_y_;
  int size_z_;
  float delta_x_;
  float offset_x_;
  float offset_y_;
  float offset_z_;
};

template <class LevelSetOp>
__global__ void SetLevelSetKernel(LevelSetBase grid, LevelSetOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < grid.SizeX() && y < grid.SizeY() && z < grid.SizeZ()) {
    float px = grid.OffsetX() + x * grid.DeltaX();
    float py = grid.OffsetY() + y * grid.DeltaX();
    float pz = grid.OffsetZ() + z * grid.DeltaX();
    grid(x, y, z) = op(px, py, pz);
  }
}

template <class LevelSetOp>
__global__ void UnionLevelSetKernel(LevelSetBase grid, LevelSetOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < grid.SizeX() && y < grid.SizeY() && z < grid.SizeZ()) {
    float px = grid.OffsetX() + x * grid.DeltaX();
    float py = grid.OffsetY() + y * grid.DeltaX();
    float pz = grid.OffsetZ() + z * grid.DeltaX();
    grid(x, y, z) = fminf(grid(x, y, z), op(px, py, pz));
  }
}

template <class LevelSetOp>
__global__ void IntersectionLevelSetKernel(LevelSetBase grid, LevelSetOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < grid.SizeX() && y < grid.SizeY() && z < grid.SizeZ()) {
    float px = grid.OffsetX() + x * grid.DeltaX();
    float py = grid.OffsetY() + y * grid.DeltaX();
    float pz = grid.OffsetZ() + z * grid.DeltaX();
    grid(x, y, z) = fmaxf(grid(x, y, z), op(px, py, pz));
  }
}

class LevelSetGrid : public LevelSetBase {
 public:
  LevelSetGrid(int size_x = 0,
               int size_y = 0,
               int size_z = 0,
               float delta_x = 0.0f,
               float offset_x = 0.0f,
               float offset_y = 0.0f,
               float offset_z = 0.0f);

  LevelSetGrid(const LevelSetGrid &other);
  LevelSetGrid(LevelSetGrid &&other) noexcept;
  LevelSetGrid &operator=(const LevelSetGrid &other);
  LevelSetGrid &operator=(LevelSetGrid &&other) noexcept;

  ~LevelSetGrid();

  std::vector<float> Download() const;

  LevelSetBase Base() {
    return {data_, size_x_, size_y_, size_z_, delta_x_, offset_x_, offset_y_, offset_z_};
  }

  const LevelSetBase Base() const {
    return {data_, size_x_, size_y_, size_z_, delta_x_, offset_x_, offset_y_, offset_z_};
  }

  template <class LevelSetOp>
  void SetLevelSet(LevelSetOp op) {
    if (size_x_ && size_y_ && size_z_) {
      dim3 block(8, 8, 8);
      dim3 grid((size_x_ + block.x - 1) / block.x, (size_y_ + block.y - 1) / block.y,
                (size_z_ + block.z - 1) / block.z);
      SetLevelSetKernel<<<grid, block>>>(Base(), op);
      cudaDeviceSynchronize();
    }
  }

  template <class LevelSetOp>
  void UnionLevelSet(LevelSetOp op) {
    if (size_x_ && size_y_ && size_z_) {
      dim3 block(8, 8, 8);
      dim3 grid((size_x_ + block.x - 1) / block.x, (size_y_ + block.y - 1) / block.y,
                (size_z_ + block.z - 1) / block.z);
      UnionLevelSetKernel<<<grid, block>>>(Base(), op);
      cudaDeviceSynchronize();
    }
  }

  template <class LevelSetOp>
  void IntersectionLevelSet(LevelSetOp op) {
    if (size_x_ && size_y_ && size_z_) {
      dim3 block(8, 8, 8);
      dim3 grid((size_x_ + block.x - 1) / block.x, (size_y_ + block.y - 1) / block.y,
                (size_z_ + block.z - 1) / block.z);
      IntersectionLevelSetKernel<<<grid, block>>>(Base(), op);
      cudaDeviceSynchronize();
    }
  }

  MeshData MarchingCubes() const;
};
