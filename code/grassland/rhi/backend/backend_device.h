#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Device;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Device)
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {

class Queue;

struct DeviceState;

class Device : public rhi::IDevice {
 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;

 protected:
  void InitializeDevice(PhysicalDevice info, Core *core, BackendAPI api = BACKEND_API_DEFAULT);
  void DestroyDevice();

 public:
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  std::string Name() const override;
  bool SupportRayTracingPipeline() const override;
  bool SupportRayQuery() const override;
  bool SupportGeometryShader() const override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  rhi::IShader *CreateShader(const ShaderDesc &desc) override;
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::IQueue *GetQueue(QueueType type) override;
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override;

 protected:
  // Called once after native device resources and all queues are initialized.
  void StartCollector();
  virtual bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const = 0;
  virtual void WaitForProgress(const std::vector<Event> &next) = 0;
  virtual void InterruptCollector() = 0;
  void ShutdownQueues();
  Core *OwnerCore() const;

 private:
  void Collect();

 protected:
  virtual DeviceState &StateDevice() = 0;
  virtual const DeviceState &StateDevice() const = 0;
};

}  // namespace grassland::rhi::backend
