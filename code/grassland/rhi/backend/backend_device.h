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
  Device(PhysicalDevice info, Core *core);
  ~Device() override;
  const PhysicalDevice &GetPhysicalDevice() const noexcept;
  std::string DeviceName() const;
  bool HasRayTracingPipeline() const;
  bool HasRayQuery() const;
  bool HasGeometryShader() const;

  virtual const void *NativeIdentity() const;

  ObjectPtr<SubmissionToken> Submit(const ObjectPtr<CommandContext> &commands,
                                    const std::vector<ObjectPtr<SubmissionToken>> &waits,
                                    ObjectPtr<SubmissionToken> completion);

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
};

}  // namespace grassland::rhi::backend
