#include "sparkium/core/data_update_tracker.h"

#include <stdexcept>

#include "grassland/graphics/frame_profile.h"

namespace sparkium {

DataUpdateTracker::~DataUpdateTracker() {
  for (auto &[address, resource] : resources_)
    resource->tracker_ = nullptr;
}

void DataUpdateTracker::Register(DataResource *resource) {
  const void *address = resource->buffer_ ? static_cast<void *>(resource->buffer_) : resource->image_;
  if (!resources_.emplace(address, resource).second)
    throw std::invalid_argument("resource already registered with DataUpdateTracker");
}

void DataUpdateTracker::Unregister(DataResource *resource) {
  const void *address = resource->buffer_ ? static_cast<void *>(resource->buffer_) : resource->image_;
  resources_.erase(address);
}

DataResource &DataUpdateTracker::Find(const void *resource) {
  auto it = resources_.find(resource);
  if (it == resources_.end())
    throw std::invalid_argument("resource is not registered with DataUpdateTracker");
  return *it->second;
}

void DataUpdateTracker::Update(graphics::Buffer *buffer, const void *data, size_t size, size_t offset) {
  static_cast<Buffer &>(Find(buffer)).Update(data, size, offset);
}

void DataUpdateTracker::Update(graphics::Image *image, const void *data) {
  static_cast<Image &>(Find(image)).Update(data);
}

void DataUpdateTracker::Update(graphics::Image *image,
                               const void *data,
                               graphics::Offset2D offset,
                               graphics::Extent2D extent) {
  static_cast<Image &>(Find(image)).Update(data, offset, extent);
}

void DataUpdateTracker::Invalidate(graphics::Buffer *buffer) {
  Find(buffer).Invalidate();
}

void DataUpdateTracker::Invalidate(graphics::Image *image) {
  Find(image).Invalidate();
}

void DataUpdateTracker::InvalidateIfTracked(graphics::Image *image) {
  auto it = resources_.find(image);
  if (it != resources_.end())
    it->second->Invalidate();
}

void DataUpdateTracker::FlushBeforeRead(graphics::Buffer *buffer) {
  if (!Find(buffer).dirty_.empty())
    Flush();
}

void DataUpdateTracker::Flush() {
  std::unique_ptr<graphics::CommandContext> commands;
  size_t copies = 0, bytes = 0;
  for (auto &[address, resource] : resources_) {
    auto &entry = *resource;
    if (entry.dirty_.empty())
      continue;
    if (!commands)
      core_->CreateCommandContext(&commands);
    std::sort(entry.dirty_.begin(), entry.dirty_.end());
    std::vector<std::pair<size_t, size_t>> ranges;
    for (auto range : entry.dirty_) {
      if (!ranges.empty() && range.first <= ranges.back().second)
        ranges.back().second = std::max(ranges.back().second, range.second);
      else
        ranges.push_back(range);
    }
    for (auto [begin, end] : ranges) {
      bytes += end - begin;
      if (entry.buffer_) {
        commands->CmdUploadBuffer(entry.buffer_, entry.bytes_.data() + begin, end - begin, begin);
        ++copies;
      } else {
        size_t pixel = graphics::PixelSize(entry.image_->Format());
        size_t pitch = entry.image_->Extent().width * pixel;
        // Full rows can share one copy; partial rows retain their exact bounds.
        while (begin < end) {
          size_t width = std::min(end - begin, pitch - begin % pitch);
          size_t rows = begin % pitch == 0 && end - begin >= pitch ? (end - begin) / pitch : 1;
          commands->CmdUploadImage(entry.image_, entry.bytes_.data() + begin,
                                   {int32_t(begin % pitch / pixel), int32_t(begin / pitch)},
                                   {uint32_t(width / pixel), uint32_t(rows)});
          begin += width * rows;
          ++copies;
        }
      }
    }
  }
  if (!commands)
    return;
  if (core_->SubmitCommandContext(commands.get()) != 0)
    throw std::runtime_error("failed to submit tracked data updates");
  for (auto &[address, resource] : resources_)
    resource->dirty_.clear();
  if (graphics::FrameProfile::active) {
    auto &counts = graphics::FrameProfile::active->counters;
    ++counts["data_update_batches"];
    counts["data_update_copies"] += copies;
    counts["data_update_bytes"] += bytes;
  }
}

}  // namespace sparkium
