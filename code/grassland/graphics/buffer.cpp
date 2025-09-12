#include "grassland/graphics/buffer.h"

namespace grassland::graphics {

BufferRange::BufferRange(Buffer *buffer, size_t offset, size_t size) : buffer(buffer), offset(offset), size(size) {
  if (buffer) {
    this->size = std::min(size, buffer->Size() - offset);
  }
}

void Buffer::PybindModuleRegistration(py::module_ &m) {
}

}  // namespace grassland::graphics
