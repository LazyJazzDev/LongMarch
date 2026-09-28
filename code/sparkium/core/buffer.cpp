#include "sparkium/core/buffer.h"

#include <cstring>
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

void Buffer::MergeUpdates() {
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

void Buffer::Invalidate() {
  capacity_ = buffer_->Size();
  ++revision_;
  bytes_.clear();
  valid_.clear();
  dirty_.clear();
}

void Buffer::RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
  MergeUpdates();
  for (auto [begin, end] : dirty_) {
    commands.CmdUploadBuffer(Get(), bytes_.data() + begin, end - begin, begin);
    bytes += end - begin;
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
  ++revision_;
}

}  // namespace sparkium
