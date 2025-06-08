#pragma once
#include "grassland/graphics/core.h"
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Buffer {
 public:
  Buffer() = default;
  virtual ~Buffer() = default;

  virtual BufferType Type() const = 0;

  virtual size_t Size() const = 0;

  virtual void Resize(size_t new_size) = 0;

  virtual void UploadData(const void *data, size_t size, size_t offset = 0) = 0;

  virtual void DownloadData(void *data, size_t size, size_t offset = 0) = 0;
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
