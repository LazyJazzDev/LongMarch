#include "sparks/core/film.h"

namespace sparks {
Film::Film(Core *core, int width, int height) : core_(core) {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &raw_image_);
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

void Film::Develop(graphics::Image *targ_img) {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram("tone_mapping"));
  cmd_context->CmdBindResources(0, {raw_image_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {targ_img}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((targ_img->Extent().width + 7) / 8, (targ_img->Extent().height + 7) / 8, 1);
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

}  // namespace sparks
