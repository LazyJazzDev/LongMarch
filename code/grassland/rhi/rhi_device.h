#pragma once
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"
#include "grassland/util/file_system.h"

namespace grassland::rhi {
class IImage;
class IBuffer;
struct BufferDesc;
class IComputeProgram;
struct ComputeProgramDesc;
class IProgram;
struct ProgramDesc;
class IShader;
struct ShaderDesc;
class IWindow;
class IQueue;
class ICommandContext;

class IDevice : public IObject {
 public:
  virtual IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) = 0;
  virtual ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) = 0;
  virtual IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) = 0;
  virtual std::string Name() const = 0;
  virtual bool SupportRayTracingPipeline() const = 0;
  virtual bool SupportRayQuery() const = 0;
  virtual bool SupportGeometryShader() const = 0;
  // Independent 2D color target, also usable for transfer/clear commands.
  virtual IImage *CreateImage(const ImageDesc &desc) = 0;
  virtual ISampler *CreateSampler(const SamplerDesc &desc = {}) = 0;
  virtual IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) = 0;
  virtual IBuffer *CreateBuffer(const BufferDesc &desc) = 0;
  virtual IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) = 0;
  virtual IProgram *CreateProgram(const ProgramDesc &desc) = 0;
  virtual IShader *CreateShader(const ShaderDesc &desc) = 0;
  IShader *CreateShader(const std::string &filename,
                        const std::string &entry_point,
                        ShaderStage stage,
                        FileSystem *fs = nullptr,
                        const std::vector<std::string> &include_paths = {},
                        const std::vector<std::pair<std::string, std::string>> &defines = {});
  IShader *CreateShaderFromSource(const std::string &source,
                                  const std::string &entry_point,
                                  ShaderStage stage,
                                  const std::vector<std::string> &include_paths = {},
                                  const std::vector<std::pair<std::string, std::string>> &defines = {});
  // No native window or input/UI callbacks. All frames share one image.
  // GetNextImage waits when three frame leases are still held by callers or commands.
  virtual IWindow *CreateHeadlessWindow(const WindowDesc &desc = {}) = 0;
  virtual IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  virtual IQueue *GetQueue(QueueType type) = 0;
  virtual ICommandContext *CreateCommandContext() = 0;

 protected:
  ~IDevice() override;
};

}  // namespace grassland::rhi
