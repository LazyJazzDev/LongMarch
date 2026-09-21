#include "sparkium/backend/cpu/cpu_buffer.h"

#include <algorithm>

namespace sparkium::backend::cpu {

CpuBuffer::CpuBuffer(size_t size, BufferType type) : type_(type) {
  memory = std::make_unique<CpuMemory>(size);
}

void CpuBuffer::Resize(size_t size) {
  auto next = std::make_unique<CpuMemory>(size);
  std::vector<uint8_t> data(std::min(size, Size()));
  memory->Download(data.data(), data.size());
  next->Upload(data.data(), data.size());
  memory = std::move(next);
}

}  // namespace sparkium::backend::cpu
