#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {

class IQueue;
class IWindow;
class ICommandContext;

class IDevice : public IObject {
 public:
  IDevice(PhysicalDevice info, const ObjectPtr<ICore> &core);
  ~IDevice() override;
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  virtual const void *NativeIdentity() const;
  virtual std::unique_ptr<IWindow> CreateWindow(const WindowDesc &desc);
  virtual std::unique_ptr<IQueue> CreateQueue(QueueType type);

  // Lightweight completion state: deliberately holds no native resources or packet.
  class SubmissionState final : public IObject {
   public:
    explicit SubmissionState(uint64_t queue_identity);
    SubmissionStatus Status() const;
    bool Wait() const;
    void Complete(SubmissionStatus status);
    uint64_t QueueIdentity() const;

   private:
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    SubmissionStatus status_{SubmissionStatus::Pending};
    uint64_t queue_identity_;
  };

  ObjectPtr<SubmissionState> Submit(const ObjectPtr<ICommandContext> &commands,
                                    const std::vector<ObjectPtr<SubmissionState>> &waits);

 protected:
  void ShutdownQueues();

 private:
  struct Submission {
    ObjectPtr<ICommandContext> commands;
    ObjectPtr<SubmissionState> completion;
  };

  void Collect();
  ObjectPtr<ICore> core_;
  PhysicalDevice info_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Submission> pending_;
  std::thread collector_;
  bool stopping_{};
  const uint64_t queue_identity_;
};

}  // namespace grassland::rhi::backend
