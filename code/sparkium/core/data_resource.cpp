#include "sparkium/core/data_resource.h"

#include <cstring>
#include <stdexcept>

#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
DataResource::DataResource(DataUpdateTracker &tracker) : tracker_(&tracker) {
}

void DataResource::MergeUpdates() {
  std::sort(dirty_.begin(), dirty_.end());
  size_t count = 0;
  for (auto range : dirty_) {
    if (count && range.first <= dirty_[count - 1].second)
      dirty_[count - 1].second = std::max(dirty_[count - 1].second, range.second);
    else
      dirty_[count++] = range;
  }
  dirty_.resize(count);
}

void DataResource::Invalidate() {
  bytes_.clear();
  valid_.clear();
  dirty_.clear();
}

void DataResource::Write(const void *data, size_t size, size_t offset) {
  if (!tracker_)
    throw std::logic_error("resource's DataUpdateTracker has been destroyed");
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
    : DataResource(tracker),
      buffer_(std::move(buffer)) {
  if (!buffer_ || buffer_->Type() != graphics::BUFFER_TYPE_STATIC)
    throw std::invalid_argument("tracked buffers require static GPU storage");
  capacity_ = buffer_->Size();
  tracker.Register(this);
}

Buffer::~Buffer() {
  if (tracker_)
    tracker_->Unregister(this);
}

void Buffer::Resize(size_t size) {
  buffer_->Resize(size);
  Invalidate();
}

void Buffer::Invalidate() {
  DataResource::Invalidate();
  capacity_ = buffer_->Size();
}

void Buffer::Update(const void *data, size_t size, size_t offset) {
  if (capacity_ != buffer_->Size())
    Invalidate();
  Write(data, size, offset);
}

Image::Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image)
    : DataResource(tracker),
      image_(std::move(image)) {
  if (!image_)
    throw std::invalid_argument("tracked images require a native image");
  capacity_ = size_t(Extent().width) * Extent().height * graphics::PixelSize(Format());
  tracker.Register(this);
}

Image::~Image() {
  if (tracker_)
    tracker_->Unregister(this);
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
