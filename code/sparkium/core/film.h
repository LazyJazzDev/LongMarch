#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Film : public Object {
 public:
  Film(Core *core, int width, int height);

  void Reset();

  Core *GetCore() const;

  glm::uvec2 GetExtent();
  int GetWidth() const;
  int GetHeight() const;

  struct Info {
    int accumulated_samples{0};
    float persistence{1.0};
    float clamping{100.0f};
    float max_exposure{16.0f};
    int view_transform{0};
    float exposure{0.0f};
    float gamma{1.0f};
    float contrast{1.0f};
  } info;

  // HDR output is linear sRGB with artistic grading and an extended Filmic look.
  void Develop(rhi::IImage *targ_image, bool linear_hdr = false);

  void RegisterResetCallback(const std::function<void()> &callback);

  rhi::IImage *GetRawImage() const;
  rhi::IImage *GetDepthImage() const;
  rhi::IImage *GetStencilImage() const;

 private:
  Core *core_;
  glm::uvec2 extent_;
  rhi::ObjPtr<rhi::IImage> raw_image_;
  rhi::ObjPtr<rhi::IImage> depth_image_;
  rhi::ObjPtr<rhi::IImage> stencil_image_;
  rhi::ObjPtr<rhi::IBuffer> tone_mapping_buffer_;

  std::vector<std::function<void()>> reset_callbacks_;
  rhi::ObjPtr<rhi::IBindingGroup> tone_mapping_group_;
};

}  // namespace sparkium
