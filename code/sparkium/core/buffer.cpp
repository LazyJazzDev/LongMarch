#include "sparkium/core/buffer.h"

#include <stdexcept>

namespace sparkium {
Buffer::Buffer(DataUpdateTracker &tracker, std::unique_ptr<graphics::Buffer> buffer)
    : tracker_(&tracker),
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
  capacity_ = buffer_->Size();
  ++revision_;
  AcknowledgeUploads();
}

size_t Buffer::PendingUploadBytes() const {
  size_t bytes = 0;
  for (const auto &task : updates_)
    bytes += task.data.size();
  return bytes;
}

void Buffer::RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
  for (const auto &task : updates_) {
    commands.CmdUploadBuffer(Get(), task.data.data(), task.data.size(), task.offset);
    bytes += task.data.size();
    ++copies;
  }
}

void Buffer::Update(const void *data, size_t size, size_t offset) {
  if (capacity_ != buffer_->Size())
    Invalidate();
  if (!tracker_)
    throw std::logic_error("resource's DataUpdateTracker has been destroyed");
  if (offset > capacity_ || size > capacity_ - offset)
    throw std::out_of_range("tracked update exceeds resource bounds");
  if (!size)
    return;
  if (!data)
    throw std::invalid_argument("null update data");
  const auto *source = static_cast<const uint8_t *>(data);
  updates_.push_back({offset, std::vector<uint8_t>(source, source + size)});
  ++revision_;
}

}  // namespace sparkium
