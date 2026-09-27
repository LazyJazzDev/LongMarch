#pragma once

#include "grassland/grassland.h"

namespace sparkium {
namespace graphics = grassland::graphics;
class DataUpdateTracker;

// Owns the CPU update state. Registration never owns the resource.
class DataResource {
 public:
  DataResource(const DataResource &) = delete;
  DataResource &operator=(const DataResource &) = delete;
  void Invalidate();

 protected:
  ~DataResource();
  void Detach();
  void Write(const void *data, size_t size, size_t offset);

 private:
  friend class DataUpdateTracker;
  friend class Buffer;
  friend class Image;
  DataResource(DataUpdateTracker &tracker, graphics::Buffer *buffer, graphics::Image *image);
  DataUpdateTracker *tracker_;
  graphics::Buffer *buffer_;
  graphics::Image *image_;
  size_t capacity_;
  std::vector<uint8_t> bytes_;
  std::vector<bool> valid_;
  std::vector<std::pair<size_t, size_t>> dirty_;
};

// Composition keeps backend objects intact for graphics commands and native casts.
class Buffer final : public DataResource {
 public:
  Buffer(DataUpdateTracker &tracker, std::unique_ptr<graphics::Buffer> buffer);
  ~Buffer();

  graphics::Buffer *Get() const {
    return buffer_.get();
  }

  size_t Size() const {
    return buffer_->Size();
  }

  graphics::BufferRange Range(size_t offset = 0, size_t size = ~0ull) const {
    return buffer_->Range(offset, size);
  }

  void Resize(size_t size);
  void Update(const void *data, size_t size, size_t offset = 0);

  void DownloadData(void *data, size_t size, size_t offset = 0) const {
    buffer_->DownloadData(data, size, offset);
  }

 private:
  std::unique_ptr<graphics::Buffer> buffer_;
};

class Image final : public DataResource {
 public:
  Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image);
  ~Image();

  graphics::Image *Get() const {
    return image_.get();
  }

  graphics::Extent2D Extent() const {
    return image_->Extent();
  }

  graphics::ImageFormat Format() const {
    return image_->Format();
  }

  void Update(const void *data);
  void Update(const void *data, graphics::Offset2D offset, graphics::Extent2D extent);

  void DownloadData(void *data) const {
    image_->DownloadData(data);
  }

 private:
  std::unique_ptr<graphics::Image> image_;
};
}  // namespace sparkium
