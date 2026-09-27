#include "sparkium/core/data_resource.h"

#include <cstring>
#include <stdexcept>

#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
DataResource::DataResource(DataUpdateTracker &tracker, graphics::Buffer *buffer, graphics::Image *image)
    : tracker_(&tracker),
      buffer_(buffer),
      image_(image) {
  if ((!buffer && !image) || (buffer && buffer->Type() != graphics::BUFFER_TYPE_STATIC))
    throw std::invalid_argument("tracked resources require an image or static buffer");
  capacity_ = buffer ? buffer->Size()
                     : size_t(image->Extent().width) * image->Extent().height * graphics::PixelSize(image->Format());
  tracker.Register(this);
}

DataResource::~DataResource() {
  Detach();
}

void DataResource::Detach() {
  if (tracker_) {
    tracker_->Unregister(this);
    tracker_ = nullptr;
  }
}

void DataResource::Invalidate() {
  bytes_.clear();
  valid_.clear();
  dirty_.clear();
  if (buffer_)
    capacity_ = buffer_->Size();
}

void DataResource::Write(const void *data, size_t size, size_t offset) {
  if (!tracker_)
    throw std::logic_error("resource's DataUpdateTracker has been destroyed");
  if (buffer_ && capacity_ != buffer_->Size())
    Invalidate();
  if (offset > capacity_ || size > capacity_ - offset)
    throw std::out_of_range("tracked update exceeds resource bounds");
  if (!size)
    return;
  if (!data)
    throw std::invalid_argument("null update data");
  const size_t end = offset + size;
  if (bytes_.size() < end) {
    bytes_.resize(end);
    valid_.resize(end, false);
  }
  if (std::all_of(valid_.begin() + offset, valid_.begin() + end, [](bool value) { return value; }) &&
      std::memcmp(bytes_.data() + offset, data, size) == 0)
    return;
  std::memcpy(bytes_.data() + offset, data, size);
  std::fill(valid_.begin() + offset, valid_.begin() + end, true);
  dirty_.emplace_back(offset, end);
}

Buffer::Buffer(DataUpdateTracker &tracker, std::unique_ptr<graphics::Buffer> buffer)
    : DataResource(tracker, buffer.get(), nullptr),
      buffer_(std::move(buffer)) {
}

Buffer::~Buffer() {
  Detach();
}

void Buffer::Resize(size_t size) {
  buffer_->Resize(size);
  Invalidate();
}

void Buffer::Update(const void *data, size_t size, size_t offset) {
  Write(data, size, offset);
}

Image::Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image)
    : DataResource(tracker, nullptr, image.get()),
      image_(std::move(image)) {
}

Image::~Image() {
  Detach();
}

void Image::Update(const void *data) {
  auto extent = Extent();
  Write(data, size_t(extent.width) * extent.height * graphics::PixelSize(Format()), 0);
}

void Image::Update(const void *data, graphics::Offset2D offset, graphics::Extent2D extent) {
  auto full = Extent();
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("tracked image region");
  if (!extent.width || !extent.height)
    return;
  if (!data)
    throw std::invalid_argument("null image update data");
  size_t pixel = graphics::PixelSize(Format());
  for (size_t y = 0; y < extent.height; ++y)
    Write(static_cast<const uint8_t *>(data) + y * extent.width * pixel, extent.width * pixel,
          ((offset.y + y) * full.width + offset.x) * pixel);
}
}  // namespace sparkium
