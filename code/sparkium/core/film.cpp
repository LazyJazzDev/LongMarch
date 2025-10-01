#include "sparkium/core/film.h"

#include "sparkium/core/core.h"

namespace sparkium {
Film::Film(Core *core, int width, int height)
    : core_(core), extent_{static_cast<uint32_t>(width), static_cast<uint32_t>(height)} {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &raw_image_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32_SINT, &stencil_image_);
}

void Film::Reset() {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdClearImage(raw_image_.get(), {1.0f, 0.0f, 1.0f, 1.0f});
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());

  for (auto &callback : reset_callbacks_) {
    callback();
  }
}

Core *Film::GetCore() const {
  return core_;
}

graphics::Extent2D Film::GetExtent() {
  return extent_;
}

int Film::GetWidth() const {
  return extent_.width;
}

int Film::GetHeight() const {
  return extent_.height;
}

void Film::Develop(graphics::Image *targ_image) {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram("tone_mapping"));
  cmd_context->CmdBindResources(0, {raw_image_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {targ_image}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((targ_image->Extent().width + 7) / 8, (targ_image->Extent().height + 7) / 8, 1);
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  core_->GraphicsCore()->WaitGPU();
}

void Film::RegisterResetCallback(const std::function<void()> &callback) {
  reset_callbacks_.push_back(callback);
}

graphics::Image *Film::GetRawImage() const {
  return raw_image_.get();
}

graphics::Image *Film::GetDepthImage() const {
  return depth_image_.get();
}

graphics::Image *Film::GetStencilImage() const {
  return stencil_image_.get();
}

}  // namespace sparkium
