#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendDevice)
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {

struct DeviceState;

class IBackendDevice : public rhi::IDevice {
 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override = 0;
  IBottomLevelAccelerationStructure *CommonCreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc);
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override = 0;
  ITopLevelAccelerationStructure *CommonCreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc);
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override = 0;
  IRayTracingProgram *CommonCreateRayTracingProgram(const RayTracingProgramDesc &desc);

 protected:
  void InitializeDevice(PhysicalDevice info, IBackendCore *core, BackendAPI api = BACKEND_API_DEFAULT);
  void DestroyDevice();

 public:
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  std::string Name() const override = 0;
  std::string CommonName() const;
  bool SupportRayTracingPipeline() const override = 0;
  bool CommonSupportRayTracingPipeline() const;
  bool SupportRayQuery() const override = 0;
  bool CommonSupportRayQuery() const;
  bool SupportGeometryShader() const override = 0;
  bool CommonSupportGeometryShader() const;
  rhi::IImage *CreateImage(const ImageDesc &desc) override = 0;
  rhi::IImage *CommonCreateImage(const ImageDesc &desc);
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override = 0;
  ISampler *CommonCreateSampler(const SamplerDesc &desc = {});
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override = 0;
  rhi::IBindingGroup *CommonCreateBindingGroup(const BindingLayoutDesc &layout);
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override = 0;
  rhi::IBuffer *CommonCreateBuffer(const BufferDesc &desc);
  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override = 0;
  rhi::IComputeProgram *CommonCreateComputeProgram(const ComputeProgramDesc &desc);
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override = 0;
  rhi::IProgram *CommonCreateProgram(const ProgramDesc &desc);
  rhi::IShader *CreateShader(const ShaderDesc &desc) override = 0;
  rhi::IShader *CommonCreateShader(const ShaderDesc &desc);
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override = 0;
  rhi::IWindow *CommonCreateGLFWWindow(const WindowDesc &desc = {});
  rhi::IQueue *GetQueue(QueueType type) override = 0;
  rhi::IQueue *CommonGetQueue(QueueType type);
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override = 0;
  bool CommonWaitEvents(const std::vector<Event> &events, bool wait_all = true) const;

 protected:
  // Called once after native device resources and all queues are initialized.
  void StartCollector();
  virtual bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const = 0;
  virtual void WaitForProgress(const std::vector<Event> &next) = 0;
  virtual void InterruptCollector() = 0;
  void ShutdownQueues();
  IBackendCore *OwnerCore() const;

 private:
  void Collect();

 protected:
  virtual DeviceState &StateDevice() = 0;
  virtual const DeviceState &StateDevice() const = 0;
};

}  // namespace grassland::rhi::backend
