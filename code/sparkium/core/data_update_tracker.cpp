#include "sparkium/core/data_update_tracker.h"

#include <stdexcept>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/core/acceleration_structure.h"

namespace sparkium {

DataUpdateTracker::~DataUpdateTracker() {
  // Remove dependents first, using the same detachment path as explicit removal.
  while (!tlases_.empty())
    Unregister(*tlases_.begin());
  while (!blases_.empty())
    Unregister(*blases_.begin());
  while (!buffers_.empty())
    Unregister(buffers_.begin()->second);
  while (!images_.empty())
    Unregister(images_.begin()->second);
}

void DataUpdateTracker::Register(Buffer *buffer) {
  if (!buffer || !buffer->IsTrackedBy(*this))
    throw std::invalid_argument("resource belongs to a different or destroyed DataUpdateTracker");
  if (!buffers_.emplace(buffer->Get(), buffer).second)
    throw std::invalid_argument("buffer already registered with DataUpdateTracker");
}

void DataUpdateTracker::Register(Image *image) {
  if (!image || !image->IsTrackedBy(*this))
    throw std::invalid_argument("resource belongs to a different or destroyed DataUpdateTracker");
  if (!images_.emplace(image->Get(), image).second)
    throw std::invalid_argument("image already registered with DataUpdateTracker");
}

void DataUpdateTracker::Unregister(Buffer *buffer) {
  if (!buffer || !buffer->IsTrackedBy(*this))
    return;
  for (auto *blas : blases_)
    blas->InvalidateBuffer(buffer->Get());
  buffer->DetachTracker();
  buffers_.erase(buffer->Get());
}

void DataUpdateTracker::Unregister(Image *image) {
  if (!image || !image->IsTrackedBy(*this))
    return;
  image->DetachTracker();
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
  if (!blas || !blas->IsTrackedBy(*this))
    throw std::invalid_argument("resource belongs to a different or destroyed DataUpdateTracker");
  if (!blases_.insert(blas).second)
    throw std::invalid_argument("BLAS already registered with DataUpdateTracker");
}

void DataUpdateTracker::Register(TopLevelAccelerationStructure *tlas) {
  if (!tlas || !tlas->IsTrackedBy(*this))
    throw std::invalid_argument("resource belongs to a different or destroyed DataUpdateTracker");
  if (!tlases_.insert(tlas).second)
    throw std::invalid_argument("TLAS already registered with DataUpdateTracker");
}

void DataUpdateTracker::Unregister(BottomLevelAccelerationStructure *blas) {
  if (!blas || !blas->IsTrackedBy(*this))
    return;
  blases_.erase(blas);
  blas->DetachTracker();
  for (auto *tlas : tlases_)
    tlas->InvalidateBLAS(blas);
}

void DataUpdateTracker::Unregister(TopLevelAccelerationStructure *tlas) {
  if (!tlas || !tlas->IsTrackedBy(*this))
    return;
  tlas->DetachTracker();
  tlases_.erase(tlas);
}

uint64_t DataUpdateTracker::Revision(graphics::Buffer *buffer) {
  return Find(buffer).Revision();
}

bool DataUpdateTracker::Contains(BottomLevelAccelerationStructure *blas) const {
  return blases_.count(blas) != 0;
}

void DataUpdateTracker::Flush() {
  std::unique_ptr<graphics::CommandContext> commands;
  size_t copies = 0, bytes = 0;
  auto record = [&](auto &resources) {
    for (auto &[address, resource] : resources) {
      if (!resource->HasUpdates())
        continue;
      if (!commands)
        core_->CreateCommandContext(&commands);
      resource->RecordUploads(*commands, copies, bytes);
    }
  };
  record(buffers_);
  record(images_);
  if (commands) {
    if (core_->SubmitCommandContext(commands.get()) != 0)
      throw std::runtime_error("failed to submit tracked data updates");
    for (auto &[address, buffer] : buffers_)
      buffer->AcknowledgeUploads();
    for (auto &[address, image] : images_)
      image->AcknowledgeUploads();
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
