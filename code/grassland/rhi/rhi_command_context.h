#pragma once
#include <array>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
class IWindow;

class ICommandContext : public IObject {
 public:
  virtual bool CmdClearColor(IImage *image, const std::array<float, 4> &color) = 0;
  // Presents the target window's acquired frame. Currently requires a preceding
  // CmdClearColor for that frame; does not copy an arbitrary image to the window.
  virtual bool CmdPresent(IWindow *window) = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
