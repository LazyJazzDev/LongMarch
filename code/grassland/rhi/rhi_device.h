#pragma once
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"
#include "grassland/rhi/rhi_timeline.h"
#include "grassland/util/file_system.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::IDevice, grassland::rhi::IObject, "7a97161c-7adc-4201-a4f0-53f40b94b527")

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
struct SubmitInfo;

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
  virtual int GetCUDADeviceIndex() const = 0;
#if defined(LONGMARCH_CUDA_RUNTIME)
  // Owned buffer exposing ICUDABuffer through QueryInterface; null if unsupported.
  virtual IBuffer *CreateCUDABuffer(const BufferDesc &desc) = 0;
  virtual ICUDATimeline *CreateCUDATimeline() = 0;
#endif
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
  virtual IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  // Returns an owned reference to a fixed device-owned queue. Repeated calls
  // return the same queue object, even after all caller references are released.
  // Main and Assist have separate timelines but may share a native queue.
  // Separate logical queues do not guarantee independent GPU progress.
  virtual IQueue *GetQueue(QueueType type) = 0;
  // Blocks the CPU until all events (or any event when wait_all is false) complete.
  // Events must belong to this device. Empty All succeeds; empty Any fails.
  virtual bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const = 0;

 protected:
  ~IDevice() override;
};

}  // namespace grassland::rhi
