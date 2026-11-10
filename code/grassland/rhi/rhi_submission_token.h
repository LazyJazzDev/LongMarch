#pragma once
#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi {

class ISubmissionToken : public IObject {
 public:
  // Tracks accepted GPU work. A Vulkan OUT_OF_DATE presentation is reported
  // through IWindow::IsSwapChainInvalid(), independently of GPU completion.
  virtual Event GetEvent() const = 0;
  virtual SubmissionStatus GetStatus() const = 0;
  virtual bool WaitHost() const = 0;

 protected:
  ~ISubmissionToken() override;
};

}  // namespace grassland::rhi
