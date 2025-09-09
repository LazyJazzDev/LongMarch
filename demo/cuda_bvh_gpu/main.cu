#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "chang_zheng.h"
#include "cub/cub.cuh"
#include "cuda_runtime.h"
#include "curand.h"

#define KERNEL_LAUNCH_SIZE(n, nthread) (n + nthread - 1) / nthread, nthread

struct CustomLess {
  LM_DEVICE_FUNC bool operator()(const float &lhs, const float &rhs) const {
    return lhs < rhs;
  }
};

__global__ void GenerateRandomNumber(float *x, int *y, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) {
    return;
  }
  curandState_t state;
  curand_init(blockIdx.x, threadIdx.x, 0, &state);
  x[idx] = curand_uniform(&state);
  y[idx] = idx;
}

int main() {
  CD::DeviceClock clock;
  const int n = 1000000;
  thrust::device_vector<float> x;
  thrust::device_vector<int> y;
  x.resize(n);
  y.resize(n);
  clock.Record("Allocate memory");
  GenerateRandomNumber<<<KERNEL_LAUNCH_SIZE(n, 256)>>>(thrust::raw_pointer_cast(x.data()),
                                                       thrust::raw_pointer_cast(y.data()), n);
  clock.Record("Generate random numbers");

  void *temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(x.data()),
                                  thrust::raw_pointer_cast(y.data()), n, CustomLess{});
  cudaMalloc(&temp_storage, temp_storage_bytes);
  clock.Record("Allocate temp storage");
  cub::DeviceMergeSort::SortPairs(temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(x.data()),
                                  thrust::raw_pointer_cast(y.data()), n, CustomLess{});
  clock.Record("Cuda merge sort");
  cudaFree(temp_storage);

  thrust::host_vector<float> x_host = x;
  thrust::host_vector<int> y_host = y;
  clock.Record("Copy to host");
  clock.Finish();

  for (int i = 0; i < 10; ++i) {
    std::cout << x_host[i * n / 10] << " " << y_host[i * n / 10] << std::endl;
  }

  return 0;
}
