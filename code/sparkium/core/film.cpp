#include "sparkium/core/film.h"

#include "sparkium/core/core.h"
#include "sparkium/core/frame_profile.h"

namespace sparkium {
Film::Film(Core *core, int width, int height)
    : core_(core),
      extent_{static_cast<uint32_t>(width), static_cast<uint32_t>(height)} {
  raw_image_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(width), static_cast<uint32_t>(height), rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  depth_image_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(width), static_cast<uint32_t>(height), rhi::IMAGE_FORMAT_D32_SFLOAT});
  stencil_image_ = core_->GetDevice()->CreateImage(
      {static_cast<uint32_t>(width), static_cast<uint32_t>(height), rhi::IMAGE_FORMAT_R32_INT});
  tone_mapping_group_ = core_->GetDevice()->CreateBindingGroup(
      core_->GetComputeProgram("tone_mapping")->GetDesc().pipeline_layout.groups[0]);
  tone_mapping_buffer_ = core_->GetDevice()->CreateBuffer({(sizeof(int) + sizeof(float) * 3 + 255) & ~uint64_t(255)});
}

void Film::Reset() {
  rhi::ObjPtr<rhi::ICommandContext> cmd_context;
  cmd_context = core_->GetDevice()->CreateCommandContext();
  cmd_context->CmdClearImage(raw_image_.Get(), {0.0f, 0.0f, 0.0f, 1.0f});
  Execute(core_->GetDevice(), cmd_context.Get());

  for (auto &callback : reset_callbacks_) {
    callback();
  }
}

Core *Film::GetCore() const {
  return core_;
}

glm::uvec2 Film::GetExtent() {
  return extent_;
}

int Film::GetWidth() const {
  return extent_.x;
}

int Film::GetHeight() const {
  return extent_.y;
}

void Film::Develop(rhi::IImage *targ_image, bool linear_hdr) {
  if (linear_hdr && targ_image->GetDesc().format != rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT)
    throw std::invalid_argument("HDR film development requires an RGBA32Float image");
  CpuProfileScope develop_profile("develop");
  rhi::ObjPtr<rhi::ICommandContext> cmd_context;
  cmd_context = core_->GetDevice()->CreateCommandContext();
  cmd_context->CmdClearImage(targ_image, {});
  cmd_context->CmdBindComputeProgram(core_->GetComputeProgram(linear_hdr ? "tone_mapping_hdr" : "tone_mapping"));
  tone_mapping_group_->BindResource(0, raw_image_.Get());
  tone_mapping_group_->BindResource(1, targ_image);
  cmd_context->CmdUploadBuffer(tone_mapping_buffer_.Get(), 0, &info.view_transform, sizeof(int) + sizeof(float) * 3);
  tone_mapping_group_->BindResource(2, tone_mapping_buffer_.Get());
  cmd_context->CmdBindGroup(0, tone_mapping_group_.Get());
  cmd_context->CmdDispatch((targ_image->GetDesc().width + 7) / 8, (targ_image->GetDesc().height + 7) / 8, 1);
  CpuProfileScope submit_profile("develop_submit_and_wait");
  Execute(core_->GetDevice(), cmd_context.Get());
  submit_profile.End();
}

void Film::RegisterResetCallback(const std::function<void()> &callback) {
  reset_callbacks_.push_back(callback);
}

rhi::IImage *Film::GetRawImage() const {
  return raw_image_.Get();
}

rhi::IImage *Film::GetDepthImage() const {
  return depth_image_.Get();
}

rhi::IImage *Film::GetStencilImage() const {
  return stencil_image_.Get();
}

}  // namespace sparkium
