#pragma once
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi {
class Window {
 public:
  // Window creation, events, acquisition and presentation run on the main thread.
  static bool PollEvents();
  bool ShouldClose() const;
  bool IsKeyDown(Key key) const;
  Image AcquireFrame() const;
  bool SetPresentMode(PresentMode mode) const;
  PresentMode GetPresentMode() const;
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::IWindow;
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
