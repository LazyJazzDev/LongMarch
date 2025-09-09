#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "chang_zheng.h"
#include "cub/cub.cuh"
#include "cuda_runtime.h"
#include "curand.h"

#define KERNEL_LAUNCH_SIZE(n, nthread) (n + nthread - 1) / nthread, nthread

__global__ void AddKernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int local_a = a[idx];
  int local_b = b[idx];
  int local_c = c[idx];
  for (int i = 0; i < 1048576; i++) {
    local_a = local_b;
    local_b = local_c;
    local_c += local_a;
  }
  a[idx] = local_a;
  b[idx] = local_b;
  c[idx] = local_c;
}

class ComputeTask {
 public:
  ComputeTask(int n_size) {
    n_size_ = n_size;
    a_.resize(n_size);
    b_.resize(n_size);
    c_.resize(n_size);
    thrust::fill(a_.begin(), a_.end(), 1);
    thrust::fill(b_.begin(), b_.end(), 2);
  }

  void SubmitTask(int n_thread, cudaStream_t stream = 0) {
    AddKernel<<<(n_size_ + n_thread - 1) / n_thread, n_thread, 0, stream>>>(
        thrust::raw_pointer_cast(a_.data()), thrust::raw_pointer_cast(b_.data()), thrust::raw_pointer_cast(c_.data()));
  }

 private:
  int n_size_;
  thrust::device_vector<int> a_;
  thrust::device_vector<int> b_;
  thrust::device_vector<int> c_;
};

int main() {
  CD::DeviceClock clock;
  constexpr int n_subtask = 4096;
  constexpr int n_task = 1024;
  ComputeTask task(n_task * n_subtask);

  std::vector<ComputeTask> tasks;
  std::vector<cudaStream_t> streams;
  for (int i = 0; i < n_task; i++) {
    tasks.emplace_back(n_subtask);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.emplace_back(stream);
  }

  CD::DeviceClock device_clock;
  task.SubmitTask(128);
  device_clock.Record("Full-concurrent task");
  for (int i = 0; i < n_task; i++) {
    tasks[i].SubmitTask(128);
  }
  device_clock.Record("Low-concurrent tasks");
  for (int i = 0; i < n_task; i++) {
    tasks[i].SubmitTask(128, streams[i]);
  }
  device_clock.Record("Stream-concurrent tasks");
  device_clock.Finish();

  for (int i = 0; i < n_task; i++) {
    cudaStreamDestroy(streams[i]);
  }

  return 0;
}
