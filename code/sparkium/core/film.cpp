#include "sparkium/core/film.h"

#include <vector>

#include "sparkium/pipelines/raytracing/cpu/cpu_shaders.h"

#include "grassland/graphics/frame_profile.h"
#include "sparkium/core/core.h"

namespace sparkium {
Film::Film(Core *core, int width, int height)
    : core_(core), extent_{static_cast<uint32_t>(width), static_cast<uint32_t>(height)} {
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &raw_image_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);
  core_->GraphicsCore()->CreateImage(width, height, graphics::IMAGE_FORMAT_R32_SINT, &stencil_image_);
  core_->GraphicsCore()->CreateBuffer(sizeof(int) + sizeof(float) * 3,
                                      graphics::BUFFER_TYPE_STATIC,
                                      &tone_mapping_buffer_);
}

void Film::Reset() {
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  cmd_context->CmdClearImage(raw_image_.get(), {0.0f, 0.0f, 0.0f, 1.0f});
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
  graphics::CpuProfileScope develop_profile("develop");
  if (core_->GraphicsCore()->API() == graphics::BACKEND_API_HOST) {
    // The host backend has no compute shaders, so the same tone mapping runs on
    // the CPU. The shader itself is unchanged; the CPU backend compiles it, so
    // both paths apply identical maths.
    const size_t pixels = static_cast<size_t>(extent_.width) * extent_.height;
    std::vector<float> raw(pixels * 4);
    raw_image_->DownloadData(raw.data());
    std::vector<uint8_t> developed(pixels * 4);
    raytracing::cpu::ToneMappingSettingsView settings;
    settings.view_transform = info.view_transform;
    settings.exposure = info.exposure;
    settings.gamma = info.gamma;
    settings.contrast = info.contrast;
    raytracing::cpu::ToneMap(raytracing::cpu::TextureView{raw.data(), extent_.width, extent_.height, 4}, developed.data(),
                             settings);
    targ_image->UploadData(developed.data());
    return;
  }
  std::unique_ptr<graphics::CommandContext> cmd_context;
  core_->GraphicsCore()->CreateCommandContext(&cmd_context);
  graphics::GpuProfileScope tone_profile(cmd_context.get(), "tone_map");
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram("tone_mapping"));
  cmd_context->CmdBindResources(0, {raw_image_.get()}, graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdBindResources(1, {targ_image}, graphics::BIND_POINT_COMPUTE);
  tone_mapping_buffer_->UploadData(&info.view_transform,
                                   sizeof(int) + sizeof(float) * 3);
  cmd_context->CmdBindResources(2, {tone_mapping_buffer_.get()},
                                graphics::BIND_POINT_COMPUTE);
  cmd_context->CmdDispatch((targ_image->Extent().width + 7) / 8, (targ_image->Extent().height + 7) / 8, 1);
  tone_profile.End();
  graphics::CpuProfileScope submit_profile("develop_submit");
  core_->GraphicsCore()->SubmitCommandContext(cmd_context.get());
  submit_profile.End();
  graphics::CpuProfileScope wait_profile("develop_wait");
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
