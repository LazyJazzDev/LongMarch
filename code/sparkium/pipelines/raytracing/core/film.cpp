#include "sparkium/core/film.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/film.h"

namespace sparkium::raytracing {

Film::Film(sparkium::Film &film) : film_(film) {
  core_ = DedicatedCast(film_.GetCore());
  film_.RegisterResetCallback([this]() { Reset(); });
  core_->GraphicsCore()->CreateImage(film_.GetWidth(), film_.GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                     &accumulated_color_);
  core_->GraphicsCore()->CreateImage(film_.GetWidth(), film_.GetHeight(), graphics::IMAGE_FORMAT_R32_SFLOAT,
                                     &accumulated_samples_);
}

void Film::Reset() {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdClearImage(accumulated_color_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdClearImage(accumulated_samples_.get(), {});
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  film_.info.accumulated_samples = 0;
}

int Film::GetWidth() const {
  return film_.GetWidth();
}

int Film::GetHeight() const {
  return film_.GetHeight();
}

Film *DedicatedCast(sparkium::Film *film) {
  COMPONENT_CAST(film, Film);
}

}  // namespace sparkium::raytracing
