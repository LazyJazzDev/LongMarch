#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
class IBuffer;
struct BufferDesc;
class IProgram;
struct ProgramDesc;
class IShader;
struct ShaderDesc;
class IWindow;
class IQueue;
class ICommandContext;

class IDevice : public IObject {
 public:
  virtual std::string Name() const = 0;
  virtual bool SupportRayTracingPipeline() const = 0;
  virtual bool SupportRayQuery() const = 0;
  virtual bool SupportGeometryShader() const = 0;
  // Independent 2D color target, also usable for transfer/clear commands.
  virtual IImage *CreateImage(const ImageDesc &desc) = 0;
  virtual ISampler *CreateSampler(const SamplerDesc &desc = {}) = 0;
  virtual IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) = 0;
  virtual IBuffer *CreateBuffer(const BufferDesc &desc) = 0;
  virtual IProgram *CreateProgram(const ProgramDesc &desc) = 0;
  virtual IShader *CreateShader(const ShaderDesc &desc) = 0;
  virtual IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  virtual IQueue *GetQueue(QueueType type) = 0;
  virtual ICommandContext *CreateCommandContext() = 0;

 protected:
  ~IDevice() override;
};

}  // namespace grassland::rhi
