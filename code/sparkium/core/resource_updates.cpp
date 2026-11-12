#include "sparkium/core/resource_updates.h"

#include <cstring>
#include <limits>

namespace sparkium {

bool ResourceUpdates::Upload(grassland::rhi::IBuffer *buffer, const void *data, uint64_t size, uint64_t offset) {
  if (!buffer || !data || !size || offset > buffer->GetDesc().size || size > buffer->GetDesc().size - offset ||
      size > std::numeric_limits<size_t>::max())
    return false;
  BufferUpload upload{grassland::rhi::ObjPtr<grassland::rhi::IBuffer>::Retain(buffer), offset,
                      std::vector<uint8_t>(static_cast<size_t>(size))};
  std::memcpy(upload.bytes.data(), data, upload.bytes.size());
  uploads_.emplace_back(std::move(upload));
  return true;
}

bool ResourceUpdates::Upload(grassland::rhi::IImage *image, const void *data, uint64_t row_pitch) {
  if (!image || !image->IsValid() || !data)
    return false;
  const auto &desc = image->GetDesc();
  if (!desc.width || !desc.height || grassland::rhi::IsDepthFormat(desc.format))
    return false;
  const uint64_t row_bytes = uint64_t(desc.width) * grassland::rhi::ImageFormatSize(desc.format);
  if (!row_pitch)
    row_pitch = row_bytes;
  if (!row_bytes || row_pitch < row_bytes || row_bytes > std::numeric_limits<size_t>::max() / desc.height ||
      row_pitch > std::numeric_limits<size_t>::max() / desc.height)
    return false;
  // Copy only texels, not caller-owned padding. Large padded rows need not stay
  // alive until rendering begins, and the staging upload remains tightly packed.
  ImageUpload upload{grassland::rhi::ObjPtr<grassland::rhi::IImage>::Retain(image), row_bytes,
                     std::vector<uint8_t>(static_cast<size_t>(row_bytes * desc.height))};
  for (uint32_t y = 0; y < desc.height; ++y)
    std::memcpy(upload.bytes.data() + y * row_bytes, static_cast<const uint8_t *>(data) + y * row_pitch, row_bytes);
  uploads_.emplace_back(std::move(upload));
  return true;
}

bool ResourceUpdates::Build(grassland::rhi::IBottomLevelAccelerationStructure *structure,
                            const std::vector<grassland::rhi::AccelerationStructureGeometryBuffers> &geometries) {
  if (!structure || geometries.empty())
    return false;
  GeometryBuild build{
      grassland::rhi::ObjPtr<grassland::rhi::IBottomLevelAccelerationStructure>::Retain(structure), geometries, {}};
  for (const auto &geometry : geometries) {
    if (!geometry.data.buffer)
      return false;
    build.buffers.push_back(grassland::rhi::ObjPtr<grassland::rhi::IBuffer>::Retain(geometry.data.buffer));
    if (geometry.indices.buffer)
      build.buffers.push_back(grassland::rhi::ObjPtr<grassland::rhi::IBuffer>::Retain(geometry.indices.buffer));
  }
  uploads_.emplace_back(std::move(build));
  return true;
}

ResourceUpdates::Cursor ResourceUpdates::Begin() const {
  return {first_, 0};
}

bool ResourceUpdates::Record(grassland::rhi::ICommandContext *commands, Cursor &cursor) const {
  if (!commands || cursor.first != first_ || cursor.count > uploads_.size())
    return false;
  for (size_t i = cursor.count; i < uploads_.size(); ++i) {
    const auto &upload = uploads_[i];
    if (const auto *buffer = std::get_if<BufferUpload>(&upload)) {
      if (!commands->CmdUploadBuffer(buffer->buffer.Get(), buffer->offset, buffer->bytes.data(), buffer->bytes.size()))
        return false;
    } else if (const auto *image = std::get_if<ImageUpload>(&upload)) {
      if (!commands->CmdUploadImage(image->image.Get(), image->bytes.data(), image->row_pitch))
        return false;
    } else {
      const auto &build = std::get<GeometryBuild>(upload);
      if (!commands->CmdBuildBottomLevelAccelerationStructure(build.structure.Get(), build.geometries))
        return false;
    }
  }
  cursor.count = uploads_.size();
  return true;
}

bool ResourceUpdates::Accept(const Cursor &cursor) {
  if (cursor.first != first_ || cursor.count > uploads_.size())
    return false;
  uploads_.erase(uploads_.begin(), uploads_.begin() + cursor.count);
  first_ += cursor.count;
  return true;
}

bool ResourceUpdates::Empty() const {
  return uploads_.empty();
}

}  // namespace sparkium
