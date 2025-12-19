#include "sparkium/core/film.h"

#include "sparkium/pipelines/raster/core/core.h"
#include "sparkium/pipelines/raster/core/film.h"

namespace sparkium::raster {

Film::Film(sparkium::Film &film) : film_(film), core_(DedicatedCast(film.GetCore())) {
  auto extent = film_.GetExtent();
  core_->GraphicsCore()->CreateImage(extent.width, extent.height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                     &albedo_roughness_buffer_);
  core_->GraphicsCore()->CreateImage(extent.width, extent.height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                     &position_specular_buffer_);
  core_->GraphicsCore()->CreateImage(extent.width, extent.height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                     &normal_metallic_buffer_);

  film_.RegisterResetCallback([this]() {
    std::unique_ptr<graphics::CommandContext> cmd_ctx;
    core_->GraphicsCore()->CreateCommandContext(&cmd_ctx);
    cmd_ctx->CmdClearImage(albedo_roughness_buffer_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
    cmd_ctx->CmdClearImage(position_specular_buffer_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
    cmd_ctx->CmdClearImage(normal_metallic_buffer_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
    core_->GraphicsCore()->SubmitCommandContext(cmd_ctx.get());
  });
}

graphics::Image *Film::GetAlbedoRoughnessBuffer() const {
  return albedo_roughness_buffer_.get();
}

graphics::Image *Film::GetPositionSpecularBuffer() const {
  return position_specular_buffer_.get();
}

graphics::Image *Film::GetNormalMetallicBuffer() const {
  return normal_metallic_buffer_.get();
}

Film *DedicatedCast(sparkium::Film *film) {
  COMPONENT_CAST(film, Film);
}

}  // namespace sparkium::raster
