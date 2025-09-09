#include "cao_di/graphics/buffer.h"

namespace CD::graphics {

BufferRange::BufferRange(Buffer *buffer, size_t offset, size_t size) : buffer(buffer), offset(offset), size(size) {
  if (buffer) {
    this->size = std::min(size, buffer->Size() - offset);
  }
}

}  // namespace CD::graphics
