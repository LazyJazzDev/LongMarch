#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi::backend {

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
