#pragma once
#include <array>
#include <atomic>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
class ITestDevice;
}
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestDevice,
                        grassland::rhi::IObject,
                        "4e3fc3ff-ce3b-487f-8223-b719d0892792")

namespace grassland::rhi::backend {
class ITestDevice : public IObject {
 public:
  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;

 protected:
  virtual bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const = 0;
  virtual void WaitForProgress(const std::vector<Event> &next) = 0;
  virtual void InterruptCollector() = 0;
};
class ITestQueue;
class ITestCommandContext;
class ITestTimeline;
class ITestImage;
class ITestWindow;
}  // namespace grassland::rhi::backend
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestQueue,
                        grassland::rhi::IObject,
                        "60afde9f-bae6-41a8-9a89-7dfe605b15ed")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestCommandContext,
                        grassland::rhi::IObject,
                        "2e409197-1d5e-4e86-8176-a4b7cbcd53cc")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestTimeline,
                        grassland::rhi::IObject,
                        "a326bd3f-4580-49cd-8121-e0218c326fdc")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestImage,
                        grassland::rhi::IObject,
                        "8eeac66e-642b-453c-91bb-03eb1136255f")
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::ITestWindow,
                        grassland::rhi::backend::IBackendWindow,
                        "4bb07dd3-6dc2-4fca-8e93-2b062a897455")

namespace grassland::rhi::backend {
class ITestQueue : public IObject {
 protected:
 public:
  virtual ITestDevice *Owner() const = 0;
  virtual ITestTimeline *GetTimeline() const = 0;
  virtual std::mutex &QueueMutex() = 0;
  virtual  // Called under QueueMutex, in native submission order.
      uint64_t
      NextSerial() = 0;
  virtual ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<ITestCommandContext> &commands,
                                                    const SubmitInfo &info) = 0;
  virtual  // Called only by the device collector; returns whether this queue is drained.
      bool
      CollectCompleted(uint64_t completed, bool timeline_valid) = 0;
};

class ITestCommandContext : public IObject {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

 protected:
 public:
  virtual void BindDownloads(rhi::ISubmissionToken *submission) = 0;
  virtual void ReleaseDownloads() = 0;
  virtual bool Seal() = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual ITestQueue *OwnerQueue() const = 0;
  virtual void MarkSubmitted() = 0;
  virtual bool WasSubmitted() const = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;

 protected:
  virtual bool EndRecording() = 0;
};

class ITestTimeline : public IObject {
 public:
  virtual ITestDevice *Owner() const = 0;
};

class ITestImage : public IObject {
 public:
  virtual bool SupportsQueue(const ITestQueue *queue) const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual const void *DeviceIdentity() const = 0;
};

class ITestWindow : public IBackendWindow {
 public:
  virtual ITestQueue *OwnerQueue() const = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;
  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual bool UpdateNextImage() = 0;
  virtual void ReleaseNextImage() = 0;
  virtual ObjectPtr<ITestImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
};
}  // namespace grassland::rhi::backend
