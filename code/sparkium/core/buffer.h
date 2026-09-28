#pragma once

#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
class Buffer final {
 public:
  Buffer(DataUpdateTracker &tracker, std::unique_ptr<graphics::Buffer> buffer);
  ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  graphics::Buffer *Get() const {
    return buffer_.get();
  }

  size_t Size() const {
    return buffer_->Size();
  }

  graphics::BufferRange Range(size_t offset = 0, size_t size = ~0ull) const {
    return buffer_->Range(offset, size);
  }

  bool IsTrackedBy(const DataUpdateTracker &tracker) const {
    return tracker_ == &tracker;
  }

  bool HasUpdates() const {
    return !dirty_.empty();
  }

  uint64_t Revision() const {
    return revision_;
  }

  void RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes);

  void Resize(size_t size);
  void Invalidate();
  void Update(const void *data, size_t size, size_t offset = 0);

  void DownloadData(void *data, size_t size, size_t offset = 0) const {
    buffer_->DownloadData(data, size, offset);
  }

 private:
  // Only successful submission and removal may acknowledge updates or detach the owner.
  friend void DataUpdateTracker::Flush();
  friend void DataUpdateTracker::Unregister(Buffer *resource);

  void DetachTracker() {
    tracker_ = nullptr;
  }

  void AcknowledgeUploads() {
    dirty_.clear();
  }

  DataUpdateTracker *tracker_;
  uint64_t revision_{};
  std::vector<uint8_t> bytes_;
  void MergeUpdates();
  size_t capacity_{};
  std::vector<bool> valid_;
  std::vector<std::pair<size_t, size_t>> dirty_;
  std::unique_ptr<graphics::Buffer> buffer_;
};
}  // namespace sparkium
