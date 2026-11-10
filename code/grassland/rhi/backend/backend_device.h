#pragma once
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

class Device : public Object<rhi::IDevice> {
 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  Device(PhysicalDevice info, Core *core, BackendAPI api = BACKEND_API_DEFAULT);
  ~Device() override;
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  int GetCUDADeviceIndex() const noexcept override;
  std::string Name() const override;
  bool SupportRayTracingPipeline() const override;
  bool SupportRayQuery() const override;
  bool SupportGeometryShader() const override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
#if defined(LONGMARCH_CUDA_RUNTIME)
  ICUDABuffer *CreateCUDABuffer(const BufferDesc &desc) override;
#endif
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
  std::array<ObjectPtr<Queue>, 2> queues_;
  std::atomic<bool> stopping_{};

 private:
  void Collect();
  BackendAPI api_;
  Core *core_;
  PhysicalDevice info_;
  std::thread collector_;
};

}  // namespace grassland::rhi::backend
