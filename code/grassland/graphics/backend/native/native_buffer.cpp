#include "native_buffer.h"

#include <algorithm>

namespace grassland::graphics::backend {

NativeBuffer::NativeBuffer(bool cuda, size_t size, BufferType type) : cuda_(cuda), type_(type) {
  memory = std::make_unique<NativeMemory>(cuda, size);
}

void NativeBuffer::Resize(size_t size) {
  auto next = std::make_unique<NativeMemory>(cuda_, size);
  std::vector<uint8_t> data(std::min(size, Size()));
  memory->Download(data.data(), data.size());
  next->Upload(data.data(), data.size());
  memory = std::move(next);
}

}  // namespace grassland::graphics::backend
