#include "sparkium/core/data_update_tracker.h"

#include <cstring>
#include <stdexcept>

#include "grassland/graphics/frame_profile.h"

namespace sparkium {

void DataUpdateTracker::Register(graphics::Buffer *buffer) {
  if (!buffer || buffer->Type() != graphics::BUFFER_TYPE_STATIC)
    throw std::invalid_argument("tracked buffers must use static GPU storage");
  auto &entry = resources_[buffer];
  if (!entry.lifetime.expired())
    return;
  entry = {};
  entry.lifetime = buffer->Lifetime();
  entry.buffer = buffer;
  entry.capacity = buffer->Size();
}

void DataUpdateTracker::Register(graphics::Image *image) {
  if (!image)
    throw std::invalid_argument("null tracked image");
  auto &entry = resources_[image];
  if (!entry.lifetime.expired())
    return;
  entry = {};
  entry.lifetime = image->Lifetime();
  entry.image = image;
  entry.capacity = size_t(image->Extent().width) * image->Extent().height * graphics::PixelSize(image->Format());
}

DataUpdateTracker::Entry &DataUpdateTracker::Find(const void *resource) {
  auto it = resources_.find(resource);
  if (it == resources_.end() || it->second.lifetime.expired())
    throw std::invalid_argument("resource is not registered with DataUpdateTracker");
  return it->second;
}

void DataUpdateTracker::Write(Entry &entry, const void *data, size_t size, size_t offset) {
  if (offset > entry.capacity || size > entry.capacity - offset)
    throw std::out_of_range("tracked update exceeds resource bounds");
  if (!size)
    return;
  if (!data)
    throw std::invalid_argument("null update data");
  const size_t end = offset + size;
  if (entry.bytes.size() < end) {
    entry.bytes.resize(end);
    entry.valid.resize(end, false);
  }
  if (std::all_of(entry.valid.begin() + offset, entry.valid.begin() + end, [](bool value) { return value; }) &&
      std::memcmp(entry.bytes.data() + offset, data, size) == 0)
    return;
  std::memcpy(entry.bytes.data() + offset, data, size);
  std::fill(entry.valid.begin() + offset, entry.valid.begin() + end, true);
  entry.dirty.emplace_back(offset, end);
}

void DataUpdateTracker::Update(graphics::Buffer *buffer, const void *data, size_t size, size_t offset) {
  auto &entry = Find(buffer);
  if (entry.capacity != buffer->Size())
    Invalidate(buffer);
  Write(entry, data, size, offset);
}

void DataUpdateTracker::Update(graphics::Image *image, const void *data) {
  auto &entry = Find(image);
  Write(entry, data, entry.capacity, 0);
}

void DataUpdateTracker::Update(graphics::Image *image,
                               const void *data,
                               graphics::Offset2D offset,
                               graphics::Extent2D extent) {
  auto &entry = Find(image);
  auto full = image->Extent();
  if (offset.x < 0 || offset.y < 0 || uint64_t(offset.x) + extent.width > full.width ||
      uint64_t(offset.y) + extent.height > full.height)
    throw std::out_of_range("tracked image region");
  if (!extent.width || !extent.height)
    return;
  if (!data)
    throw std::invalid_argument("null image update data");
  size_t pixel = graphics::PixelSize(image->Format());
  for (size_t y = 0; y < extent.height; ++y)
    Write(entry, static_cast<const uint8_t *>(data) + y * extent.width * pixel, extent.width * pixel,
          ((offset.y + y) * full.width + offset.x) * pixel);
}

void DataUpdateTracker::Invalidate(graphics::Buffer *buffer) {
  auto &entry = Find(buffer);
  entry.bytes.clear();
  entry.valid.clear();
  entry.dirty.clear();
  entry.capacity = buffer->Size();
}

void DataUpdateTracker::Invalidate(graphics::Image *image) {
  auto &entry = Find(image);
  entry.bytes.clear();
  entry.valid.clear();
  entry.dirty.clear();
}

void DataUpdateTracker::FlushBeforeRead(graphics::Buffer *buffer) {
  if (!Find(buffer).dirty.empty())
    Flush();
}

void DataUpdateTracker::Flush() {
  std::unique_ptr<graphics::CommandContext> commands;
  size_t copies = 0, bytes = 0;
  for (auto it = resources_.begin(); it != resources_.end();) {
    if (it->second.lifetime.expired()) {
      it = resources_.erase(it);
      continue;
    }
    auto &entry = it++->second;
    if (entry.dirty.empty())
      continue;
    if (!commands)
      core_->CreateCommandContext(&commands);
    std::sort(entry.dirty.begin(), entry.dirty.end());
    std::vector<std::pair<size_t, size_t>> ranges;
    for (auto range : entry.dirty) {
      if (!ranges.empty() && range.first <= ranges.back().second)
        ranges.back().second = std::max(ranges.back().second, range.second);
      else
        ranges.push_back(range);
    }
    for (auto [begin, end] : ranges) {
      bytes += end - begin;
      if (entry.buffer) {
        commands->CmdUploadBuffer(entry.buffer, entry.bytes.data() + begin, end - begin, begin);
        ++copies;
      } else {
        size_t pixel = graphics::PixelSize(entry.image->Format());
        size_t pitch = entry.image->Extent().width * pixel;
        // Full rows can share one copy; partial rows retain their exact bounds.
        while (begin < end) {
          size_t width = std::min(end - begin, pitch - begin % pitch);
          size_t rows = begin % pitch == 0 && end - begin >= pitch ? (end - begin) / pitch : 1;
          commands->CmdUploadImage(entry.image, entry.bytes.data() + begin,
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
  for (auto &[resource, entry] : resources_)
    entry.dirty.clear();
  if (graphics::FrameProfile::active) {
    auto &counts = graphics::FrameProfile::active->counters;
    ++counts["data_update_batches"];
    counts["data_update_copies"] += copies;
    counts["data_update_bytes"] += bytes;
  }
}

}  // namespace sparkium
