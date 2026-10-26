#include "sparkium/core/film.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/film.h"

namespace sparkium::raytracing {

Film::Film(sparkium::Film &film) : film_(film) {
  core_ = DedicatedCast(film_.GetCore());
  film_.RegisterResetCallback([this]() { Reset(); });
  accumulated_color_ = core_->GetDevice()->CreateImage({static_cast<uint32_t>(film_.GetWidth()),
                                                        static_cast<uint32_t>(film_.GetHeight()),
                                                        rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  accumulated_samples_ =
      core_->GetDevice()->CreateImage({static_cast<uint32_t>(film_.GetWidth()),
                                       static_cast<uint32_t>(film_.GetHeight()), rhi::IMAGE_FORMAT_R32_SFLOAT});
  film_.Reset();
}

void Film::Reset() {
  rhi::ObjPtr<rhi::ICommandContext> cmd_context;
  cmd_context = core_->GetDevice()->CreateCommandContext();
  cmd_context->CmdClearImage(accumulated_color_.Get(), {0.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdClearImage(accumulated_samples_.Get(), {});
  Execute(core_->GetDevice(), cmd_context.Get());
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
