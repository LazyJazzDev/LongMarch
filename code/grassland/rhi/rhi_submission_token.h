#pragma once
#include "grassland/rhi/rhi_event.h"

namespace grassland::rhi {

class ISubmissionToken : public IObject {
 public:
  // Tracks accepted GPU work. A Vulkan OUT_OF_DATE presentation is reported
  // through IWindow::IsSwapChainInvalid(), independently of GPU completion.
  virtual SubmissionStatus GetStatus() const = 0;
  // Also waits for collector retirement and result publication.
  virtual bool WaitHost() const = 0;
  // Returns one owned reference to the immutable GPU completion point.
  virtual IEvent *GetEvent() const = 0;

 protected:
  ~ISubmissionToken() override;
};

}  // namespace grassland::rhi
