#pragma once

#include "sparkium/core/data_update_tracker.h"

namespace sparkium {
class Image final {
 public:
  Image(DataUpdateTracker &tracker, std::unique_ptr<graphics::Image> image);
  ~Image();
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;

  graphics::Image *Get() const {
    return image_.get();
  }

  graphics::Extent2D Extent() const {
    return image_->Extent();
  }

  graphics::ImageFormat Format() const {
    return image_->Format();
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

  void Invalidate();
  void Update(const void *data);
  void Update(const void *data, graphics::Offset2D offset, graphics::Extent2D extent);

  void DownloadData(void *data) const {
    image_->DownloadData(data);
  }

 private:
  // Only successful submission and removal may acknowledge updates or detach the owner.
  friend void DataUpdateTracker::Flush();
  friend void DataUpdateTracker::Unregister(Image *resource);

  void DetachTracker() {
    tracker_ = nullptr;
  }

  void AcknowledgeUploads() {
    dirty_.clear();
  }

  DataUpdateTracker *tracker_;
  uint64_t revision_{};
  std::vector<uint8_t> bytes_;

  // Half-open pixel bounds, independent of image row pitch and pixel byte size.
  struct Region {
    uint32_t left, top, right, bottom;
  };

  static void AddRegion(std::vector<Region> &regions, Region region);
  bool HasSnapshot(Region region) const;
  std::vector<Region> valid_;
  std::vector<Region> dirty_;
  std::unique_ptr<graphics::Image> image_;
};
}  // namespace sparkium
