#include "xing_huo/core/film.h"

namespace XH {
Film::Film(Core *core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &accumulated_color_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32_SFLOAT, &accumulated_samples_);
}

void Film::Reset() {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdClearImage(accumulated_color_.get(), {0.0f, 0.0f, 0.0f, 0.0f});
  cmd_context->CmdClearImage(accumulated_samples_.get(), {});
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  info.accumulated_samples = 0;
}

int Film::GetWidth() const {
  return accumulated_color_->Extent().width;
}

int Film::GetHeight() const {
  return accumulated_color_->Extent().height;
}

}  // namespace XH
