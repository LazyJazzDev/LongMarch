#include "sparkium/core/image.h"

#include <stdexcept>

namespace sparkium {
Image::Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image)
    : tracker_(&tracker),
      image_(std::move(image)) {
  if (!image_)
    throw std::invalid_argument("tracked images require a native image");
  tracker.Register(this);
}

Image::~Image() {
  if (tracker_)
    tracker_->Unregister(this);
}

void Image::Invalidate() {
  ++revision_;
  AcknowledgeUploads();
}

size_t Image::PendingUploadBytes() const {
  size_t bytes = 0;
  for (const auto &task : updates_)
    bytes += task.data.size();
  return bytes;
}

void Image::Update(const void *data) {
  Update(data, {0, 0}, Extent());
}

void Image::Update(const void *data, graphics::Offset2D offset, graphics::Extent2D extent) {
  if (!tracker_)
    throw std::logic_error("image is no longer registered with a DataUpdateTracker");
  auto full = Extent();
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("tracked image region");
  if (!extent.width || !extent.height)
    return;
  if (!data)
    throw std::invalid_argument("null image update data");
  const size_t size = size_t(extent.width) * extent.height * graphics::PixelSize(Format());
  const auto *source = static_cast<const uint8_t *>(data);
  updates_.push_back({offset, extent, std::vector<uint8_t>(source, source + size)});
  ++revision_;
}

void Image::RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
  // Keep submission order for last-write-wins overlaps; each task is one 2D copy.
  for (const auto &task : updates_) {
    commands.CmdUploadImage(Get(), task.data.data(), task.offset, task.extent);
    bytes += task.data.size();
    ++copies;
  }
}
}  // namespace sparkium
