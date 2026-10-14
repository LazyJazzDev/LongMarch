#pragma once
#include "grassland/rhi/rhi_image.h"
#include "grassland/rhi/rhi_object_container.h"

namespace grassland::rhi {
class Window : public ObjectContainer {
 public:
  Window() noexcept;
  // Window creation, events, acquisition and presentation run on the main thread.
  static bool PollEvents();
  bool ShouldClose() const;
  bool IsKeyDown(Key key) const;
  bool SetTitle(const std::string &title) const;
  Image AcquireFrame() const;
  bool SetPresentMode(PresentMode mode) const;
  PresentMode GetPresentMode() const;

 private:
  using Interface = backend::IWindow;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
