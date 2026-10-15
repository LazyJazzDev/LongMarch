#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend {

class Queue;
class Window;
class CommandContext;

class Device : public Object<rhi::IDevice> {
 public:
  Device(PhysicalDevice info, Core *core);
  ~Device() override;
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  std::string Name() const override;
  bool SupportRayTracingPipeline() const override;
  bool SupportRayQuery() const override;
  bool SupportGeometryShader() const override;
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::IQueue *GetQueue(QueueType type) override;
  rhi::ICommandContext *CreateCommandContext() override;
  virtual const void *NativeIdentity() const;
  virtual std::unique_ptr<Window> CreateNativeWindow(const WindowDesc &desc);
  virtual std::unique_ptr<CommandContext> CreateNativeCommandContext();
  virtual std::unique_ptr<Queue> CreateQueue(QueueType type);
  ObjectPtr<Window> ManageWindow(std::unique_ptr<Window> window);

  // Lightweight completion state: deliberately holds no native resources or packet.
  class SubmissionState final : public Object<rhi::ISubmissionToken> {
   public:
    explicit SubmissionState(uint64_t queue_identity);
    SubmissionStatus GetStatus() const override;
    bool WaitHost() const override;
    SubmissionStatus Status() const;
    bool Wait() const;
    void Complete(SubmissionStatus status);
    uint64_t QueueIdentity() const;

   private:
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    SubmissionStatus status_{SUBMISSION_STATUS_PENDING};
    uint64_t queue_identity_;
  };

  ObjectPtr<SubmissionState> Submit(const ObjectPtr<CommandContext> &commands,
                                    const std::vector<ObjectPtr<SubmissionState>> &waits);

 protected:
  void ShutdownQueues();
  Core *OwnerCore() const;

 private:
  struct Submission {
    ObjectPtr<CommandContext> commands;
    ObjectPtr<SubmissionState> completion;
  };

  void Collect();
  void CollectWindows();
  std::mutex windows_mutex_;
  std::vector<ObjectPtr<Window>> windows_;
  Core *core_;
  PhysicalDevice info_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Submission> pending_;
  std::thread collector_;
  bool stopping_{};
  const uint64_t queue_identity_;
};

}  // namespace grassland::rhi::backend
