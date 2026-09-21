#include "sparkium/backend/cuda/cuda_memory.h"

#include <algorithm>
#include <cstring>

#include "sparkium/backend/cuda/cuda_util.h"
#include "sparkium/backend/cuda/device_memory.h"

namespace sparkium::backend::cuda {

CudaMemory::CudaMemory(size_t size) : size_(size) {
  data_ = AllocateCudaMemory(std::max(size, size_t(16)));
}

CudaMemory::~CudaMemory() {
  FreeCudaMemory(data_);
}

void CudaMemory::Upload(const void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("compute buffer upload");
  if (!size)
    return;
  UploadCudaMemory(data_, p, size, offset);
}

void CudaMemory::Download(void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("compute buffer download");
  if (!size)
    return;
  DownloadCudaMemory(p, data_, size, offset);
}

}  // namespace sparkium::backend::cuda
