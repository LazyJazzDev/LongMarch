#include "sparkium/backend/cuda/cuda_buffer.h"

#include <algorithm>

namespace sparkium::backend::cuda {

CudaBuffer::CudaBuffer(size_t size, BufferType type) : type_(type) {
  memory = std::make_unique<CudaMemory>(size);
}

void CudaBuffer::Resize(size_t size) {
  auto next = std::make_unique<CudaMemory>(size);
  std::vector<uint8_t> data(std::min(size, Size()));
  memory->Download(data.data(), data.size());
  next->Upload(data.data(), data.size());
  memory = std::move(next);
}

}  // namespace sparkium::backend::cuda
