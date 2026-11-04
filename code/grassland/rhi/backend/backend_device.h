#pragma once
#include <array>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_submission.h"
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
  virtual const void *NativeIdentity() const;
  Queue *QueueState(QueueType type) const;
  bool WaitEvents(const std::vector<IEvent *> &events, bool wait_all = true) const override;
  SubmissionStatus QueryEvent(TimelinePoint point) const;
  bool HasNativeTimelines() const;
  bool QueryTimeline(QueueType type, uint64_t &value) const;
  virtual bool WaitTimelines(const std::vector<TimelinePoint> &points, bool wait_all) const;
  virtual void WakeCollector();

  ObjectPtr<SubmissionToken> Submit(const ObjectPtr<CommandContext> &commands,
                                    const std::vector<ObjectPtr<Event>> &waits,
                                    Queue *queue);

 protected:
  void ShutdownQueues();
  void InitializeQueue(ObjectPtr<Queue> queue);
  void ReleaseQueues();
  Core *OwnerCore() const;

 private:
  void Collect();
  void CollectTimelines();
  BackendAPI api_;
  Core *core_;
  PhysicalDevice info_;
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  std::array<ObjectPtr<Queue>, 2> queues_;
  std::list<Submission> pending_;
  std::vector<TimelinePoint> collector_points_;
  std::thread collector_;
  bool stopping_{};
};

}  // namespace grassland::rhi::backend
