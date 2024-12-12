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

  void UploadData(const void *data, size_t size);

  void DownloadData(void *data, size_t size);

  virtual void UploadData(const void *data, size_t size, size_t offset) = 0;

  virtual void DownloadData(void *data, size_t size, size_t offset) = 0;
};

}  // namespace grassland::graphics
