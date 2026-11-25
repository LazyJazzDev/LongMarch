#pragma once
#include <array>

#include "grassland/rhi/rhi.h"

namespace rhi_mip_demo {
// Used by the interactive demo and the offscreen rendering regression test.
class Scene {
 public:
  bool Initialize(grassland::rhi::IDevice *device, grassland::rhi::ImageFormat target_format);
  bool Draw(grassland::rhi::ICommandContext *commands, grassland::rhi::IImage *target, float seconds);

  uint32_t Anisotropy() const {
    return anisotropy_;
  }

 private:
  grassland::rhi::IDevice *device_{};
  grassland::rhi::ObjPtr<grassland::rhi::IProgram> program_;
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> vertices_, indices_, uniforms_;
  grassland::rhi::ObjPtr<grassland::rhi::IImage> texture_, depth_;
  std::array<grassland::rhi::ObjPtr<grassland::rhi::ISampler>, 3> samplers_;
  std::array<grassland::rhi::BindingGroup, 9> groups_;
  uint32_t anisotropy_{};
};
}  // namespace rhi_mip_demo
