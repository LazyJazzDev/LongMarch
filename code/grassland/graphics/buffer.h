#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

struct BufferRange {
  Buffer *buffer;
  size_t offset;
  size_t size;
  explicit BufferRange(Buffer *buffer = nullptr, size_t offset = 0, size_t size = ~0ull);
};

class Buffer {
 public:
  Buffer() = default;
  virtual ~Buffer() = default;

  virtual BufferType Type() const = 0;

  virtual size_t Size() const = 0;

  virtual void Resize(size_t new_size) = 0;

  virtual void UploadData(const void *data, size_t size, size_t offset = 0) = 0;

  virtual void DownloadData(void *data, size_t size, size_t offset = 0) = 0;

  BufferRange Range(size_t offset = 0, size_t size = ~0ull) {
    return BufferRange(this, offset, size);
  }

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<Buffer> &c);
#endif
};

#if defined(LONGMARCH_CUDA_RUNTIME)
class CUDABuffer : virtual public Buffer {
 public:
  virtual void GetCUDAMemoryPointer(void **ptr) = 0;

  template <typename T>
  void GetCUDAMemoryPointer(T **ptr) {
    void *raw_ptr = nullptr;
    GetCUDAMemoryPointer(&raw_ptr);
    *ptr = static_cast<T *>(raw_ptr);
  }
};
#endif

}  // namespace grassland::graphics
