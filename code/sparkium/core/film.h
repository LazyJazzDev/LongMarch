#pragma once
#include "core.h"
#include "sparkium/core/core_util.h"

namespace sparkium {

class Film : public Object {
 public:
  Film(Core *core, int width, int height);

  void Reset();

  Core *GetCore() const;

  graphics::Extent2D GetExtent();
  int GetWidth() const;
  int GetHeight() const;

  struct Info {
    int accumulated_samples{0};
    float persistence{1.0};
    float clamping{100.0f};
    float max_exposure{1.0f};
  } info;

  void Develop(graphics::Image *targ_image);

  void RegisterResetCallback(const std::function<void()> &callback);

  graphics::Image *GetRawImage() const;
  graphics::Image *GetDepthImage() const;
  graphics::Image *GetStencilImage() const;

 private:
  Core *core_;
  graphics::Extent2D extent_;
  std::unique_ptr<graphics::Image> raw_image_;
  std::unique_ptr<graphics::Image> depth_image_;
  std::unique_ptr<graphics::Image> stencil_image_;

  std::vector<std::function<void()>> reset_callbacks_;
};

}  // namespace sparkium
