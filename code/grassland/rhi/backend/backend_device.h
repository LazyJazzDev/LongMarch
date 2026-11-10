#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {

class CommandContext;

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
  rhi::ICommandContext *CreateCommandContext() override;
  virtual const void *NativeIdentity() const;

  ObjectPtr<SubmissionToken> Submit(const ObjectPtr<CommandContext> &commands,
                                    const std::vector<ObjectPtr<SubmissionToken>> &waits);

 protected:
  void ShutdownQueues();
  Core *OwnerCore() const;

 private:
  struct Submission {
    ObjectPtr<CommandContext> commands;
    ObjectPtr<SubmissionToken> completion;
  };

  void Collect();
  BackendAPI api_;
  Core *core_;
  PhysicalDevice info_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Submission> pending_;
  std::thread collector_;
  bool stopping_{};
};

}  // namespace grassland::rhi::backend
