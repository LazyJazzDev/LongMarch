#pragma once

#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
// Owns the CPU update state. Registration never owns the resource.
class DataResource {
 public:
  DataResource(const DataResource &) = delete;
  DataResource &operator=(const DataResource &) = delete;
  void Invalidate();

  bool IsTrackedBy(const DataUpdateTracker &tracker) const {
    return tracker_ == &tracker;
  }

  bool HasUpdates() const {
    return !dirty_.empty();
  }

  uint64_t Revision() const {
    return revision_;
  }

 protected:
  explicit DataResource(DataUpdateTracker &tracker);
  ~DataResource() = default;

  DataUpdateTracker *Tracker() const {
    return tracker_;
  }

  size_t Capacity() const {
    return capacity_;
  }

  void ResetCapacity(size_t capacity);
  void RecordUploads(graphics::CommandContext &commands, graphics::Buffer *buffer, size_t &copies, size_t &bytes);
  void RecordUploads(graphics::CommandContext &commands, graphics::Image *image, size_t &copies, size_t &bytes);
  void Write(const void *data, size_t size, size_t offset);

 private:
  // Submission acknowledgement and lifetime detachment must remain tracker-controlled.
  friend void DataUpdateTracker::Flush();
  friend void DataUpdateTracker::Unregister(Buffer *buffer);
  friend void DataUpdateTracker::Unregister(Image *image);

  void DetachTracker() {
    tracker_ = nullptr;
  }

  void AcknowledgeUploads() {
    dirty_.clear();
  }

  void MergeUpdates();
  DataUpdateTracker *tracker_;
  size_t capacity_{};
  uint64_t revision_{};
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

  void RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
    DataResource::RecordUploads(commands, Get(), copies, bytes);
  }

  void Resize(size_t size);
  void Invalidate();
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

  void RecordUploads(graphics::CommandContext &commands, size_t &copies, size_t &bytes) {
    DataResource::RecordUploads(commands, Get(), copies, bytes);
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
