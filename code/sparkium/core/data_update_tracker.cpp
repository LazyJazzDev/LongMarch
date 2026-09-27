#include "sparkium/core/data_update_tracker.h"

#include <stdexcept>

#include "grassland/graphics/frame_profile.h"

namespace sparkium {

DataUpdateTracker::~DataUpdateTracker() {
  for (auto *blas : blases_)
    blas->tracker_ = nullptr;
  for (auto *tlas : tlases_)
    tlas->tracker_ = nullptr;
  for (auto &[address, buffer] : buffers_)
    buffer->tracker_ = nullptr;
  for (auto &[address, image] : images_)
    image->tracker_ = nullptr;
}

void DataUpdateTracker::Register(Buffer *buffer) {
  if (!buffers_.emplace(buffer->Get(), buffer).second)
    throw std::invalid_argument("buffer already registered with DataUpdateTracker");
}

void DataUpdateTracker::Register(Image *image) {
  if (!images_.emplace(image->Get(), image).second)
    throw std::invalid_argument("image already registered with DataUpdateTracker");
}

void DataUpdateTracker::Unregister(Buffer *buffer) {
  for (auto *blas : blases_) {
    if (blas->vertices_.buffer == buffer->Get())
      blas->vertices_.buffer = nullptr;
    if (blas->indices_.buffer == buffer->Get())
      blas->indices_.buffer = nullptr;
  }
  buffers_.erase(buffer->Get());
}

void DataUpdateTracker::Unregister(Image *image) {
  images_.erase(image->Get());
}

Buffer &DataUpdateTracker::Find(graphics::Buffer *buffer) {
  auto it = buffers_.find(buffer);
  if (it == buffers_.end())
    throw std::invalid_argument("buffer is not registered with DataUpdateTracker");
  return *it->second;
}

Image &DataUpdateTracker::Find(graphics::Image *image) {
  auto it = images_.find(image);
  if (it == images_.end())
    throw std::invalid_argument("image is not registered with DataUpdateTracker");
  return *it->second;
}

void DataUpdateTracker::Update(graphics::Buffer *buffer, const void *data, size_t size, size_t offset) {
  Find(buffer).Update(data, size, offset);
}

void DataUpdateTracker::Update(graphics::Image *image, const void *data) {
  Find(image).Update(data);
}

void DataUpdateTracker::Update(graphics::Image *image,
                               const void *data,
                               graphics::Offset2D offset,
                               graphics::Extent2D extent) {
  Find(image).Update(data, offset, extent);
}

void DataUpdateTracker::Invalidate(graphics::Buffer *buffer) {
  Find(buffer).Invalidate();
}

void DataUpdateTracker::Invalidate(graphics::Image *image) {
  Find(image).Invalidate();
}

void DataUpdateTracker::InvalidateIfTracked(graphics::Image *image) {
  auto it = images_.find(image);
  if (it != images_.end())
    it->second->Invalidate();
}

void DataUpdateTracker::Register(BottomLevelAccelerationStructure *blas) {
  blases_.insert(blas);
}

void DataUpdateTracker::Register(TopLevelAccelerationStructure *tlas) {
  tlases_.insert(tlas);
}

void DataUpdateTracker::Unregister(BottomLevelAccelerationStructure *blas) {
  blases_.erase(blas);
  for (auto *tlas : tlases_)
    for (auto &instance : tlas->instances_)
      if (instance.blas == blas) {
        instance.blas = nullptr;
        tlas->dirty_ = true;
      }
}

void DataUpdateTracker::Unregister(TopLevelAccelerationStructure *tlas) {
  tlases_.erase(tlas);
}

void DataUpdateTracker::Flush() {
  std::unique_ptr<graphics::CommandContext> commands;
  size_t copies = 0, bytes = 0;
  auto prepare = [&](DataResource &resource) {
    if (resource.dirty_.empty())
      return false;
    if (!commands)
      core_->CreateCommandContext(&commands);
    resource.MergeUpdates();
    return true;
  };
  for (auto &[address, buffer] : buffers_) {
    if (!prepare(*buffer))
      continue;
    for (auto [begin, end] : buffer->dirty_) {
      commands->CmdUploadBuffer(address, buffer->bytes_.data() + begin, end - begin, begin);
      bytes += end - begin;
      ++copies;
    }
  }
  for (auto &[address, image] : images_) {
    if (!prepare(*image))
      continue;
    size_t pixel = graphics::PixelSize(image->Format());
    size_t pitch = image->Extent().width * pixel;
    for (auto [begin, end] : image->dirty_) {
      bytes += end - begin;
      // Full rows can share one copy; partial rows retain their exact bounds.
      while (begin < end) {
        size_t width = std::min(end - begin, pitch - begin % pitch);
        size_t rows = begin % pitch == 0 && end - begin >= pitch ? (end - begin) / pitch : 1;
        commands->CmdUploadImage(address, image->bytes_.data() + begin,
                                 {int32_t(begin % pitch / pixel), int32_t(begin / pitch)},
                                 {uint32_t(width / pixel), uint32_t(rows)});
        begin += width * rows;
        ++copies;
      }
    }
  }
  if (commands) {
    if (core_->SubmitCommandContext(commands.get()) != 0)
      throw std::runtime_error("failed to submit tracked data updates");
    for (auto &[address, buffer] : buffers_)
      buffer->dirty_.clear();
    for (auto &[address, image] : images_)
      image->dirty_.clear();
    if (graphics::FrameProfile::active) {
      auto &counts = graphics::FrameProfile::active->counters;
      ++counts["data_update_batches"];
      counts["data_update_copies"] += copies;
      counts["data_update_bytes"] += bytes;
    }
  }
  for (auto *blas : blases_)
    blas->Build();
  for (auto *tlas : tlases_)
    tlas->Build();
}

}  // namespace sparkium
