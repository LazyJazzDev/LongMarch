#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendDevice)

namespace grassland::rhi::backend {

class IBackendDevice : public rhi::IDevice {
 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override = 0;

  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override = 0;

  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override = 0;

  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;

  std::string Name() const override = 0;

  bool SupportRayTracingPipeline() const override = 0;

  bool SupportRayQuery() const override = 0;

  bool SupportGeometryShader() const override = 0;

  rhi::IImage *CreateImage(const ImageDesc &desc) override = 0;

  ISampler *CreateSampler(const SamplerDesc &desc = {}) override = 0;

  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override = 0;

  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override = 0;

  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override = 0;

  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override = 0;

  rhi::IShader *CreateShader(const ShaderDesc &desc) override = 0;

  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override = 0;

  rhi::IQueue *GetQueue(QueueType type) override = 0;

  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override = 0;

 protected:
  virtual bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const = 0;

  virtual void WaitForProgress(const std::vector<Event> &next) = 0;

  virtual void InterruptCollector() = 0;
};

}  // namespace grassland::rhi::backend
