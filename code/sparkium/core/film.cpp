#include "sparkium/core/film.h"

#include "sparkium/core/core.h"
#include "sparkium/core/render_layout.h"

namespace sparkium {
Film::Film(Core *core, int width, int height) : core_(core), extent_(width, height) {
  if (!core || !core->IsValid() || width <= 0 || height <= 0)
    return;
  auto *device = core->GetDevice();
  raw_image_ = device->CreateImage({extent_.x, extent_.y, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  depth_image_ = device->CreateImage({extent_.x, extent_.y, rhi::IMAGE_FORMAT_D32_SFLOAT});
  stencil_image_ = device->CreateImage({extent_.x, extent_.y, rhi::IMAGE_FORMAT_R32_INT});
  tone_mapping_buffer_ = device->CreateBuffer({256});
  auto *program = core->GetComputeProgram("tone_mapping");
  if (program)
    tone_mapping_group_ = device->CreateBindingGroup(program->GetDesc().pipeline_layout.groups[0]);
  if (!IsValid())
    core->ReportFailure("Cannot create Film resources");
}

bool Film::IsValid() const {
  return core_ && core_->IsValid() && raw_image_ && depth_image_ && stencil_image_ && tone_mapping_buffer_ &&
         tone_mapping_group_;
}

bool Film::PrepareAccumulation() {
  if (accumulation_group_)
    return true;
  if (!IsValid())
    return false;
  auto *device = core_->GetDevice();
  rhi::ObjPtr<rhi::IImage> color = device->CreateImage({extent_.x, extent_.y, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT});
  rhi::ObjPtr<rhi::IImage> samples = device->CreateImage({extent_.x, extent_.y, rhi::IMAGE_FORMAT_R32_SFLOAT});
  rhi::ObjPtr<rhi::IBindingGroup> group =
      device->CreateBindingGroup(RayTracingLayout(false, 1, 1, 1).groups[SPARKIUM_GROUP_FILM]);
  if (!color || !samples || !group || !group->BindResource(SPARKIUM_FILM_COLOR, color.Get()) ||
      !group->BindResource(SPARKIUM_FILM_SAMPLES, samples.Get()))
    return false;
  auto *program = core_->GetComputeProgram("film2img");
  if (!program)
    return false;
  rhi::ObjPtr<rhi::IBindingGroup> resolve = device->CreateBindingGroup(program->GetDesc().pipeline_layout.groups[0]);
  if (!resolve || !resolve->BindResource(0, color.Get()) || !resolve->BindResource(1, samples.Get()) ||
      !resolve->BindResource(2, raw_image_.Get()))
    return false;
  accumulated_color_ = std::move(color);
  accumulated_samples_ = std::move(samples);
  accumulation_group_ = std::move(group);
  resolve_group_ = std::move(resolve);
  Reset();
  return true;
}

void Film::Reset() {
  reset_pending_ = true;
  info.accumulated_samples = 0;
  for (auto &callback : reset_callbacks_)
    callback();
}

bool Film::RecordReset(rhi::ICommandContext *commands) {
  if (!commands || !IsValid())
    return false;
  if (!reset_pending_)
    return true;
  if (!commands->CmdClearImage(raw_image_.Get(), {0.0f, 0.0f, 0.0f, 1.0f}))
    return false;
  return !accumulated_color_ || (commands->CmdClearImage(accumulated_color_.Get(), {}) &&
                                 commands->CmdClearImage(accumulated_samples_.Get(), {}));
}

bool Film::RecordResolve(rhi::ICommandContext *commands) {
  return commands && resolve_group_ && commands->CmdBindComputeProgram(core_->GetComputeProgram("film2img")) &&
         commands->CmdBindGroup(0, resolve_group_.Get()) &&
         commands->CmdDispatch((extent_.x + 7) / 8, (extent_.y + 7) / 8, 1);
}

bool Film::RecordDevelop(rhi::ICommandContext *commands, rhi::IImage *target, bool linear_hdr) {
  if (!commands || !IsValid() || !target || target->GetDesc().width != extent_.x ||
      target->GetDesc().height != extent_.y ||
      target->GetDesc().format !=
          (linear_hdr ? rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT : rhi::IMAGE_FORMAT_R8G8B8A8_UNORM))
    return false;
  return commands->CmdUploadBuffer(tone_mapping_buffer_.Get(), 0, &info.view_transform,
                                   sizeof(int) + sizeof(float) * 3) &&
         tone_mapping_group_->BindResource(0, raw_image_.Get()) && tone_mapping_group_->BindResource(1, target) &&
         tone_mapping_group_->BindResource(2, tone_mapping_buffer_.Get()) &&
         commands->CmdBindComputeProgram(core_->GetComputeProgram(linear_hdr ? "tone_mapping_hdr" : "tone_mapping")) &&
         commands->CmdBindGroup(0, tone_mapping_group_.Get()) &&
         commands->CmdDispatch((extent_.x + 7) / 8, (extent_.y + 7) / 8, 1);
}

void Film::AcceptRender(int samples) {
  reset_pending_ = false;
  info.accumulated_samples += samples;
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

rhi::IImage *Film::GetRawImage() const {
  return raw_image_.Get();
}

rhi::IImage *Film::GetDepthImage() const {
  return depth_image_.Get();
}

rhi::IImage *Film::GetStencilImage() const {
  return stencil_image_.Get();
}

rhi::IBindingGroup *Film::AccumulationGroup() const {
  return accumulation_group_.Get();
}

void Film::RegisterResetCallback(const std::function<void()> &callback) {
  reset_callbacks_.push_back(callback);
}

}  // namespace sparkium
