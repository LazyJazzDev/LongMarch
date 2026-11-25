#pragma once
#include "grassland/rhi/rhi.h"

namespace rhi_pcf_demo {
class Scene {
 public:
  bool Initialize(grassland::rhi::IDevice *device, grassland::rhi::ImageFormat target_format);
  bool Draw(grassland::rhi::ICommandContext *commands, grassland::rhi::IImage *target, bool visibility_only = false);

 private:
  grassland::rhi::IDevice *device_{};
  grassland::rhi::ObjPtr<grassland::rhi::IProgram> shadow_program_, scene_program_;
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> vertices_, uniforms_;
  grassland::rhi::ObjPtr<grassland::rhi::IImage> shadow_, depth_;
  grassland::rhi::ObjPtr<grassland::rhi::ISampler> hard_sampler_, pcf_sampler_;
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> shadow_group_;
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> scene_group_;
  uint32_t shadow_vertices_{}, scene_vertices_{};
};
}  // namespace rhi_pcf_demo
