#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IWindow;
class IQueue;
class ICommandContext;

class IDevice : public IObject {
 public:
  virtual std::string Name() const = 0;
  virtual bool SupportRayTracingPipeline() const = 0;
  virtual bool SupportRayQuery() const = 0;
  virtual bool SupportGeometryShader() const = 0;
  virtual IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  virtual IQueue *GetQueue(QueueType type) = 0;
  virtual ICommandContext *CreateCommandContext() = 0;

 protected:
  ~IDevice() override;
};

}  // namespace grassland::rhi
