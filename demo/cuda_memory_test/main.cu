#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "curand.h"
#include "iostream"
#include "thrust/device_vector.h"

__global__ void MemoryTestKernel() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

template <int block_size>
struct MemoryBlock {
  uint32_t data[block_size >> 2]{};
  __host__ __device__ MemoryBlock() = default;

  __host__ __device__ MemoryBlock<block_size> operator+(const MemoryBlock<block_size> &other) const {
    MemoryBlock<block_size> result;
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  __host__ __device__ MemoryBlock<block_size> operator-(const MemoryBlock<block_size> &other) const {
    MemoryBlock<block_size> result;
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  __host__ __device__ MemoryBlock<block_size> operator*(const MemoryBlock<block_size> &other) const {
    MemoryBlock<block_size> result;
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      result.data[i] = data[i] * other.data[i];
    }
    return result;
  }

  __host__ __device__ MemoryBlock<block_size> operator^(const MemoryBlock<block_size> &other) const {
    MemoryBlock<block_size> result;
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      result.data[i] = data[i] ^ other.data[i];
    }
    return result;
  }

  __host__ __device__ MemoryBlock<block_size> operator+=(const MemoryBlock<block_size> &other) {
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      data[i] += other.data[i];
    }
    return *this;
  }

  __host__ __device__ MemoryBlock<block_size> operator-=(const MemoryBlock<block_size> &other) {
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      data[i] -= other.data[i];
    }
    return *this;
  }

  __host__ __device__ MemoryBlock<block_size> operator*=(const MemoryBlock<block_size> &other) {
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      data[i] *= other.data[i];
    }
    return *this;
  }

  __host__ __device__ MemoryBlock<block_size> operator^=(const MemoryBlock<block_size> &other) {
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      data[i] ^= other.data[i];
    }
    return *this;
  }

  __host__ __device__ bool operator==(const MemoryBlock<block_size> &other) const {
#pragma unroll
    for (int i = 0; i < block_size >> 2; i++) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }
};

template <int block_size>
__global__ void GenerateRandomBlocksKernel(MemoryBlock<block_size> *data, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }
  curandState state{};
  curand_init(0, idx, 0, &state);
  for (int i = 0; i < block_size >> 2; i++) {
    data[idx].data[i] = curand(&state);
  }
}

template <int block_size>
__global__ void TestRandomAccessThroughputKernel(const MemoryBlock<block_size> *data,
                                                 MemoryBlock<block_size> *result,
                                                 int num_elements,
                                                 int num_access) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  MemoryBlock<block_size> sum{};
  curandState state{};
  //  curand_init(0, idx, 0, &state);
  curand_init(0, blockIdx.x, 0, &state);

#pragma unroll
  for (int i = 0; i < num_access; i++) {
    // rand a int in [0, num_elements - 1]
    uint32_t idy = curand(&state) % num_elements;
    MemoryBlock<block_size> other = data[(idx + idy) % num_elements];
    sum ^= other;
  }
  result[idx] = sum;
}

template <int block_size>
void TestRandomAccessThroughput(int num_elements, int num_access) {
  uint64_t total_memory_load = static_cast<uint64_t>(num_elements) * sizeof(MemoryBlock<block_size>) * num_access;
  std::cout << "========================================================\n";
  std::cout << "Testing Random Access - Block Size " << sizeof(MemoryBlock<block_size>) << " Num Elements "
            << num_elements << " Num Access " << num_access << std::endl;
  thrust::device_vector<MemoryBlock<block_size>> data(num_elements);
  thrust::device_vector<MemoryBlock<block_size>> result(num_elements);
  float max_throughput = 0;
  float min_throughput = 0;
  float sum_throughput = 0;
  float elapsed_time = 0;
  for (int i = 0; i < 5; i++) {
    GenerateRandomBlocksKernel<block_size>
        <<<num_elements / 256 + 1, 256>>>(thrust::raw_pointer_cast(data.data()), num_elements);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    TestRandomAccessThroughputKernel<block_size><<<num_elements / 256 + 1, 256>>>(
        thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(result.data()), num_elements, num_access);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= 1000;
    float throughput = static_cast<float>(total_memory_load) / elapsed_time / 1024 / 1024 / 1024;
    if (i == 0) {
      max_throughput = throughput;
      min_throughput = throughput;
    } else {
      max_throughput = std::max(max_throughput, throughput);
      min_throughput = std::min(min_throughput, throughput);
    }
    sum_throughput += throughput;
  }

  std::cout << "Throughput: max - " << max_throughput << "GB/s min - " << min_throughput << "GB/s mean - "
            << sum_throughput / 5 << "GB/s" << std::endl;
}

template <int block_size>
__global__ void TestSequentialAccessThroughputKernel(const MemoryBlock<block_size> *data,
                                                     MemoryBlock<block_size> *result,
                                                     int num_elements,
                                                     int num_access) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  MemoryBlock<block_size> sum{};
  curandState state{};
  curand_init(0, idx, 0, &state);
  uint32_t idy = curand(&state) % num_elements;
  curand_init(0, blockIdx.x, 0, &state);
  uint32_t idz = curand(&state) % num_elements;

#pragma unroll
  for (int i = 0; i < num_access; i++) {
    // rand a int in [0, num_elements - 1]
    MemoryBlock<block_size> other = data[(idx ^ i) % num_elements];
    sum ^= other;
  }
  result[idx] = sum;
}

template <int block_size>
void TestSequentialAccessThroughput(int num_elements, int num_access) {
  uint64_t total_memory_load = static_cast<uint64_t>(num_elements) * sizeof(MemoryBlock<block_size>) * num_access;
  std::cout << "========================================================\n";
  std::cout << "Testing Sequential Access - Block Size " << sizeof(MemoryBlock<block_size>) << " Num Elements "
            << num_elements << " Num Access " << num_access << std::endl;
  thrust::device_vector<MemoryBlock<block_size>> data(num_elements);
  thrust::device_vector<MemoryBlock<block_size>> result(num_elements);
  float max_throughput = 0;
  float min_throughput = 0;
  float sum_throughput = 0;
  float elapsed_time = 0;
  for (int i = 0; i < 5; i++) {
    GenerateRandomBlocksKernel<block_size>
        <<<num_elements / 256 + 1, 256>>>(thrust::raw_pointer_cast(data.data()), num_elements);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    TestSequentialAccessThroughputKernel<block_size><<<num_elements / 256 + 1, 256>>>(
        thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(result.data()), num_elements, num_access);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= 1000;
    float throughput = static_cast<float>(total_memory_load) / elapsed_time / 1024 / 1024 / 1024;
    if (i == 0) {
      max_throughput = throughput;
      min_throughput = throughput;
    } else {
      max_throughput = std::max(max_throughput, throughput);
      min_throughput = std::min(min_throughput, throughput);
    }
    sum_throughput += throughput;
  }

  std::cout << "Throughput: max - " << max_throughput << "GB/s min - " << min_throughput << "GB/s mean - "
            << sum_throughput / 5 << "GB/s" << std::endl;
}

int main() {
  TestRandomAccessThroughput<8>(1024 * 1024, 8192);
  TestRandomAccessThroughput<16>(1024 * 1024, 4096);
  TestRandomAccessThroughput<32>(1024 * 1024, 2048);
  TestRandomAccessThroughput<64>(1024 * 1024, 1024);
  TestRandomAccessThroughput<128>(1024 * 1024, 512);
  TestRandomAccessThroughput<256>(1024 * 1024, 256);
  TestRandomAccessThroughput<512>(1024 * 1024, 128);
  TestRandomAccessThroughput<1024>(1024 * 1024, 64);
  TestRandomAccessThroughput<8>(1024 * 1024 * 128, 64);
  TestRandomAccessThroughput<16>(1024 * 1024 * 64, 64);
  TestRandomAccessThroughput<32>(1024 * 1024 * 32, 64);
  TestRandomAccessThroughput<64>(1024 * 1024 * 16, 64);
  TestRandomAccessThroughput<128>(1024 * 1024 * 8, 64);
  TestRandomAccessThroughput<256>(1024 * 1024 * 4, 64);
  TestRandomAccessThroughput<512>(1024 * 1024 * 2, 64);
  TestRandomAccessThroughput<1024>(1024 * 1024, 64);

  TestSequentialAccessThroughput<8>(1024 * 1024, 8192);
  TestSequentialAccessThroughput<16>(1024 * 1024, 4096);
  TestSequentialAccessThroughput<32>(1024 * 1024, 2048);
  TestSequentialAccessThroughput<64>(1024 * 1024, 1024);
  TestSequentialAccessThroughput<128>(1024 * 1024, 512);
  TestSequentialAccessThroughput<256>(1024 * 1024, 256);
  TestSequentialAccessThroughput<512>(1024 * 1024, 128);
  TestSequentialAccessThroughput<1024>(1024 * 1024, 64);
  TestSequentialAccessThroughput<8>(1024 * 1024 * 128, 64);
  TestSequentialAccessThroughput<16>(1024 * 1024 * 64, 64);
  TestSequentialAccessThroughput<32>(1024 * 1024 * 32, 64);
  TestSequentialAccessThroughput<64>(1024 * 1024 * 16, 64);
  TestSequentialAccessThroughput<128>(1024 * 1024 * 8, 64);
  TestSequentialAccessThroughput<256>(1024 * 1024 * 4, 64);
  TestSequentialAccessThroughput<512>(1024 * 1024 * 2, 64);
  TestSequentialAccessThroughput<1024>(1024 * 1024, 64);
  return 0;
}
