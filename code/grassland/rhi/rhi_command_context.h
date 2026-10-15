#pragma once
#include <array>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
class IWindow;

class ICommandContext : public IObject {
 public:
  // Cmd calls encode native commands immediately and retain the exact resources
  // they use. Parameter values are consumed during the call, not during Submit.
  // Submit ends recording; a context cannot be recorded or submitted again.
  virtual bool CmdClearColor(IImage *image, const std::array<float, 4> &color) = 0;
  // Presents the target window's acquired frame. Currently requires a preceding
  // CmdClearColor for that frame; does not copy an arbitrary image to the window.
  virtual bool CmdPresent(IWindow *window) = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
