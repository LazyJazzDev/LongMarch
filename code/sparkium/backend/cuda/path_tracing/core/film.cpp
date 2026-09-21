#include "sparkium/core/film.h"

#include "sparkium/backend/cuda/path_tracing/core/core.h"
#include "sparkium/backend/cuda/path_tracing/core/film.h"

namespace sparkium::cuda_tracing {

Film::Film(sparkium::Film &film) : film_(film) {
  core_ = DedicatedCast(film_.GetCore());
  film_.RegisterResetCallback([this]() { Reset(); });
  core_->BackendDevice()->CreateImage(film_.GetWidth(), film_.GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                      &accumulated_color_);
  core_->BackendDevice()->CreateImage(film_.GetWidth(), film_.GetHeight(), graphics::IMAGE_FORMAT_R32_SFLOAT,
                                      &accumulated_samples_);
  Reset();
}

void Film::Reset() {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->BackendDevice()->CreateCommandContext(&cmd_context);
  cmd_context->CmdClearImage(accumulated_color_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdClearImage(accumulated_samples_.get(), {});
  core_->BackendDevice()->SubmitCommandContext(cmd_context.get());
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

}  // namespace sparkium::cuda_tracing
