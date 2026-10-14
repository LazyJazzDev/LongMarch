#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
// Window operations and presentation run on the main thread.
bool PollEvents();

class IWindow : public IObject {
 public:
  virtual bool ShouldClose() const = 0;
  virtual bool IsKeyDown(Key key) const = 0;
  virtual bool SetTitle(const std::string &title) = 0;
  virtual IImage *AcquireFrame() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;

 protected:
  ~IWindow() override;
};

}  // namespace grassland::rhi
