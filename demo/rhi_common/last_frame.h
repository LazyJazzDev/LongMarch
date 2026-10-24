#pragma once
#include <string>

#include "grassland/rhi/rhi.h"

namespace rhi_demo {
// Enabled captures copy each rendered frame so closing a window at any time
// can still save its last completed contents, including the ImGui overlay.
class LastFrame {
 public:
  LastFrame(const std::string &directory, const std::string &name);
  bool Record(grassland::rhi::ICommandContext *commands,
              grassland::rhi::IImage *image,
              grassland::rhi::IWindow *window = nullptr);
  bool Save();

 private:
  std::string directory_, name_;
  grassland::rhi::ObjPtr<grassland::rhi::IDownloadToken> download_;
  grassland::rhi::ImageDesc desc_{};
  grassland::rhi::ColorSpace space_{grassland::rhi::COLOR_SPACE_SRGB_NONLINEAR};
  float white_scale_{1};
};
}  // namespace rhi_demo
