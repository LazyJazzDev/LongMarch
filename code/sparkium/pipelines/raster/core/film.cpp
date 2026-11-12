#include "sparkium/core/film.h"

#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"

namespace sparkium::raster {

Film::Film(sparkium::Film &film) : film_(film), core_(DedicatedCast(film.GetCore())) {
  auto extent = film_.GetExtent();
  albedo_roughness_buffer_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(extent.x), static_cast<uint32_t>(extent.y), rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  position_specular_buffer_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(extent.x), static_cast<uint32_t>(extent.y), rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  normal_metallic_buffer_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(extent.x), static_cast<uint32_t>(extent.y), rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
}

bool Film::RecordReset(rhi::ICommandContext *commands) {
  return commands && commands->CmdClearImage(albedo_roughness_buffer_.Get(), {}) &&
         commands->CmdClearImage(position_specular_buffer_.Get(), {}) &&
         commands->CmdClearImage(normal_metallic_buffer_.Get(), {}) &&
         commands->CmdClearImage(film_.GetStencilImage(),
                                 rhi::ClearValue{int32_t(0), int32_t(0), int32_t(0), int32_t(0)});
}

rhi::IImage *Film::GetAlbedoRoughnessBuffer() const {
  return albedo_roughness_buffer_.Get();
}

rhi::IImage *Film::GetPositionSpecularBuffer() const {
  return position_specular_buffer_.Get();
}

rhi::IImage *Film::GetNormalMetallicBuffer() const {
  return normal_metallic_buffer_.Get();
}

Film *DedicatedCast(sparkium::Film *film) {
  COMPONENT_CAST(film, Film);
}

}  // namespace sparkium::raster
