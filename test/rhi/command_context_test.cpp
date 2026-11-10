#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <type_traits>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhix/rhix.h"

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;

namespace {
using backend::ObjectPtr;

struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool complete{};
  uint64_t completed_limit{UINT64_MAX};
  int destroyed{};
  int resources_destroyed{};
  int devices_destroyed{};
  int windows_destroyed{};
};

class TestWindow final : public backend::Window {
 public:
  explicit TestWindow(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestWindow() override {
    EXPECT_EQ(gate_->devices_destroyed, 0);
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->windows_destroyed;
  }

  ObjectPtr<backend::Image> AcquireNativeFrame() override {
    return {};
  }

  bool UpdateNextImage() override {
    return true;
  }

  bool RebuildNativeSwapChain(PresentMode) override {
    return true;
  }

  PresentMode GetPresentMode() const override {
    return PRESENT_MODE_VSYNC;
  }

  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override {
    return {{PRESENT_MODE_VSYNC, IMAGE_FORMAT_R8G8B8A8_UNORM}};
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestImage final : public backend::Image {
 public:
  explicit TestImage(std::shared_ptr<Gate> gate, backend::Window *window = nullptr)
      : backend::Image({16, 16, IMAGE_FORMAT_R8G8B8A8_UNORM}),
        gate_(std::move(gate)),
        window_(window) {
  }

  ~TestImage() override {
    if (window_)
      EXPECT_EQ(gate_->windows_destroyed, 0);
    {
      std::lock_guard<std::mutex> lock(gate_->mutex);
      ++gate_->destroyed;
    }
    gate_->changed.notify_all();
  }

  bool IsValid() const override {
    return true;
  }

  const void *DeviceIdentity() const override {
    return gate_.get();
  }

 private:
  std::shared_ptr<Gate> gate_;
  backend::Window *window_;
};

class TestResource final : public backend::Object<rhi::IObject> {
 public:
  explicit TestResource(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestResource() override {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->resources_destroyed;
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestTimeline final : public backend::Timeline {
 public:
  TestTimeline(backend::Device *device, std::shared_ptr<Gate> gate) : Timeline(device), gate_(std::move(gate)) {
  }

  bool GetCompletedValue(uint64_t &value) const override {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    value = gate_->complete ? std::min(value_, gate_->completed_limit) : 0;
    return true;
  }

  bool WaitHost(uint64_t value) const override {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->changed.wait(lock, [&] { return !value || (gate_->complete && value_ >= value); });
    return true;
  }

  void Signal(uint64_t value) {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    value_ = value;
    gate_->changed.notify_all();
  }

 private:
  uint64_t value_{};
  std::shared_ptr<Gate> gate_;
};

class TestCommandContext final : public backend::CommandContext {
 public:
  TestCommandContext(rhi::IQueue *queue, std::shared_ptr<Gate> gate, bool fail_after_submit = false)
      : backend::CommandContext(static_cast<backend::Queue *>(queue)),
        gate_(std::move(gate)),
        fail_after_submit_(fail_after_submit) {
  }

  std::vector<ClearValue> recorded_values;
  int recorded_presents{};
  int ended{};

  bool CmdUseResource(rhi::IObject *resource) {
    return RetainResource(resource);
  }

  bool CmdClearImage(rhi::IImage *image, const ClearValue &value) override {
#if !defined(NDEBUG)
    if (!image) {
      recording_failed_ = true;
      return false;
    }
#endif
    if (!RetainResource(image))
      return false;
    recorded_values.push_back(value);
    return true;
  }

  bool CmdPresent(rhi::IWindow *window) override {
#if !defined(NDEBUG)
    if (sealed_ || !window)
      return false;
#endif
    if (recording_failed_)
      return false;
    ++recorded_presents;
    return true;
  }

  bool EndRecording() override {
    ++ended;
    return true;
  }

  bool UpdatePresentedImages() override {
    return true;
  }

  SubmitResult SubmitNative(const SubmitInfo &, Event event) override {
    static_cast<TestTimeline *>(event.timeline)->Signal(event.value);
    EXPECT_EQ(ended, 1);
    if (fail_after_submit_) {
      MarkSubmitted();
      throw std::bad_alloc{};
    }
    return SUBMIT_RESULT_SUBMITTED;
  }

  SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) override {
    if (!timeline_valid)
      return SUBMISSION_STATUS_FAILED;
    return completed >= signal.value ? SUBMISSION_STATUS_COMPLETED : SUBMISSION_STATUS_PENDING;
  }

 private:
  std::shared_ptr<Gate> gate_;
  bool fail_after_submit_;
};

class TestQueue final : public backend::Queue {
 public:
  TestTimeline *GetTimeline() const override {
    return timeline_.get();
  }

  rhi::ICommandContext *CreateCommandContext() override {
    return new TestCommandContext(this, gate_);
  }

  TestQueue(backend::Device *device, QueueType type, std::shared_ptr<Gate> gate)
      : backend::Queue(device, type),
        gate_(std::move(gate)),
        timeline_(std::make_unique<TestTimeline>(device, gate_)) {
  }

 private:
  std::shared_ptr<Gate> gate_;
  std::unique_ptr<TestTimeline> timeline_;
};

class TestDevice final : public backend::Device {
 public:
  explicit TestDevice(std::shared_ptr<Gate> gate) : backend::Device({}, {}), gate_(std::move(gate)) {
    for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST})
      queues_[type] = ObjectPtr<backend::Queue>::Adopt(new TestQueue(this, type, gate_));
    StartCollector();
  }

  ~TestDevice() override {
    {
      std::lock_guard<std::mutex> lock(gate_->mutex);
      gate_->complete = true;
      gate_->completed_limit = UINT64_MAX;
    }
    gate_->changed.notify_all();
    ShutdownQueues();
    ++gate_->devices_destroyed;
  }

 private:
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const override {
    if (wait_all) {
      for (auto event : events)
        if (!event.timeline->WaitHost(event.value))
          return false;
      return true;
    }
    for (;;) {
      for (auto event : events) {
        uint64_t value;
        if (event.timeline->GetCompletedValue(value) && value >= event.value)
          return true;
      }
      std::unique_lock<std::mutex> lock(gate_->mutex);
      gate_->changed.wait_for(lock, std::chrono::milliseconds(10));
    }
  }

  void WaitForProgress(const std::vector<Event> &next) override {
    for (auto event : next) {
      uint64_t value;
      if (event.timeline->GetCompletedValue(value) && value >= event.value)
        return;
    }
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->changed.wait_for(lock, std::chrono::milliseconds(10));
  }

  std::shared_ptr<Gate> gate_;
};

TEST(RhiCommands, CommandsAreEncodedImmediatelyAndKeepParameterValues) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  auto *native = new TestCommandContext(queue.Get(), gate);
  CommandContext context = native;
  ClearValue color{1.0f, 0.0f, 0.0f, 1.0f};
  ASSERT_TRUE(context.CmdClearImage(image, color));
  ASSERT_EQ(native->recorded_values.size(), 1u);
  color = {0.0f, 1.0f, 0.0f, 1.0f};
  ASSERT_TRUE(context.CmdClearImage(image, color));
  ASSERT_EQ(native->recorded_values.size(), 2u);
  EXPECT_EQ(native->recorded_values[0].color.float32, (std::array<float, 4>{1, 0, 0, 1}));
  EXPECT_EQ(native->recorded_values[1].color.float32, color.color.float32);
  EXPECT_EQ(native->ended, 0);
  image.Release();
  EXPECT_EQ(gate->destroyed, 0);
  ASSERT_TRUE(context.CmdPresent(window));
  EXPECT_EQ(native->recorded_presents, 1);
  auto token = std::move(context).Submit();
  ASSERT_TRUE(token);
  EXPECT_EQ(native->ended, 1);
  EXPECT_EQ(native->recorded_values.size(), 2u);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(token.WaitHost());
  EXPECT_EQ(gate->destroyed, 1);
}

TEST(RhiCommands, DifferentResourceTypesAreRetainedOnceUntilSubmissionCompletes) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto window = Window(new TestWindow(gate));
  auto image = ObjectPtr<TestImage>::Adopt(new TestImage(gate));
  auto resource = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto another = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto *native = new TestCommandContext(queue.Get(), gate);
  CommandContext context = native;
  ASSERT_TRUE(native->CmdUseResource(resource.Get()));
  ASSERT_TRUE(native->CmdClearImage(image.Get(), {0.0f, 0.0f, 1.0f, 1.0f}));
  ASSERT_TRUE(native->CmdUseResource(another.Get()));
  ASSERT_TRUE(native->CmdUseResource(resource.Get()));
  ASSERT_TRUE(native->CmdUseResource(image.Get()));
  auto *borrowed_image = image.Get();
  auto *borrowed_resource = resource.Get();
  auto *borrowed_another = another.Get();
  image.Reset();
  resource.Reset();
  another.Reset();
  EXPECT_TRUE(borrowed_image->IsUniquelyOwned());
  EXPECT_TRUE(borrowed_resource->IsUniquelyOwned());
  EXPECT_TRUE(borrowed_another->IsUniquelyOwned());
  ASSERT_TRUE(context.CmdPresent(window));
  auto token = std::move(context).Submit();
  ASSERT_TRUE(token);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 0);
    EXPECT_EQ(gate->resources_destroyed, 0);
    gate->complete = true;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(token.WaitHost());
  EXPECT_EQ(gate->destroyed, 1);
  EXPECT_EQ(gate->resources_destroyed, 2);
}

TEST(RhiCommands, UnsubmittedContextReleasesAllResourceTypes) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto context = ObjectPtr<TestCommandContext>::Adopt(new TestCommandContext(queue.Get(), gate));
  auto image = ObjectPtr<TestImage>::Adopt(new TestImage(gate));
  auto resource = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  ASSERT_TRUE(context->CmdUseResource(image.Get()));
  ASSERT_TRUE(context->CmdUseResource(resource.Get()));
  image.Reset();
  resource.Reset();
  EXPECT_EQ(gate->destroyed, 0);
  EXPECT_EQ(gate->resources_destroyed, 0);
  context.Reset();
  EXPECT_EQ(gate->destroyed, 1);
  EXPECT_EQ(gate->resources_destroyed, 1);
}

TEST(RhiCommands, RecordingAndSubmissionKeepExactImageWithoutKeepingPayloadInToken) {
  static_assert(!std::is_copy_constructible_v<CommandContext>);
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto context = CommandContext(new TestCommandContext(queue.Get(), gate));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearImage(image, {0.0f, 0.0f, 1.0f, 1.0f}));
  ASSERT_TRUE(context.CmdPresent(window));
  image = {};
  auto recorded = std::move(context);
  ASSERT_FALSE(context);
  ASSERT_TRUE(recorded);
  auto token = std::move(recorded).Submit();
  ASSERT_FALSE(recorded);
  ASSERT_TRUE(token);
  EXPECT_EQ(token.GetStatus(), SUBMISSION_STATUS_PENDING);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 0);
    gate->complete = true;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(token.WaitHost());
  EXPECT_EQ(gate->destroyed, 1);  // The still-live token does not retain the image.
  token.Release();
  queue.Release();
  window.Release();
  device.Reset();
  EXPECT_EQ(gate->devices_destroyed, 1);
  EXPECT_FALSE(token);
}

TEST(RhiCommands, CompletionSnapshotRetiresOnlyValuesAtOrBelowItsLimit) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  std::vector<SubmissionToken> tokens;
  for (int i = 0; i < 3; ++i) {
    auto context = CommandContext(new TestCommandContext(queue.Get(), gate));
    auto image = Image(new TestImage(gate));
    ASSERT_TRUE(context.CmdClearImage(image, {}));
    tokens.push_back(std::move(context).Submit());
    ASSERT_TRUE(tokens.back());
  }
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->completed_limit = 2;
    gate->complete = true;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(tokens[1].WaitHost());
  EXPECT_EQ(tokens[0].GetStatus(), SUBMISSION_STATUS_COMPLETED);
  EXPECT_EQ(tokens[2].GetStatus(), SUBMISSION_STATUS_PENDING);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 2);
    gate->completed_limit = 3;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(tokens[2].WaitHost());
  EXPECT_EQ(gate->destroyed, 3);
}

TEST(RhiCommands, DroppedTokenStillRetiresSingleSubmissionAutomatically) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto context = CommandContext(new TestCommandContext(queue.Get(), gate));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearImage(image, {1.0f, 0.0f, 0.0f, 1.0f}));
  ASSERT_TRUE(context.CmdPresent(window));
  ASSERT_TRUE(std::move(context).Submit());
  image = {};
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  std::unique_lock<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->destroyed == 1; }));
}

#if !defined(NDEBUG)
TEST(RhiCommands, CancellationReleasesResources) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto image = Image(new TestImage(gate));
  auto context = queue.CreateCommandContext();
  ASSERT_TRUE(context.CmdClearImage(image, {1.0f, 0.0f, 0.0f, 1.0f}));
  image.Release();
  EXPECT_EQ(gate->destroyed, 0);
  context.Release();
  EXPECT_EQ(gate->destroyed, 1);
}

TEST(RhiCommands, SealedContextsCannotRecordOrSubmitAgain) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto image = Image(new TestImage(gate));
  auto native = ObjectPtr<TestCommandContext>::Adopt(new TestCommandContext(queue.Get(), gate));
  auto context = CommandContext(ObjPtr<rhi::ICommandContext>::Retain(native.Get()).Detach());
  ASSERT_TRUE(context.CmdClearImage(image, {1.0f, 0.0f, 0.0f, 1.0f}));
  auto token = std::move(context).Submit();
  ASSERT_TRUE(token);
  EXPECT_FALSE(native->CmdClearImage(image.Get(), {0.0f, 1.0f, 0.0f, 1.0f}));
  EXPECT_EQ(native->Submit(), nullptr);
  native.Reset();
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  EXPECT_TRUE(token.WaitHost());
}

#endif

TEST(RhiCommands, CrossQueueDependencyDoesNotWaitForGpuOnCpu) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto assist = Queue(device->GetQueue(QUEUE_TYPE_ASSIST));
  auto main = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto context = CommandContext(new TestCommandContext(assist.Get(), gate));
  auto upload = std::move(context).Submit();
  ASSERT_TRUE(upload);
  context = CommandContext(new TestCommandContext(main.Get(), gate));
  auto done = std::move(context).Submit({{upload.GetEvent()}});
  ASSERT_TRUE(done);
  EXPECT_EQ(done.GetStatus(), SUBMISSION_STATUS_PENDING);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  EXPECT_TRUE(done.WaitHost());
}

TEST(RhiCommands, AllocationFailureAfterNativeSubmitKeepsResourcesUntilCompletion) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto context = CommandContext(new TestCommandContext(queue.Get(), gate, true));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearImage(image, {0.0f, 0.0f, 1.0f, 1.0f}));
  ASSERT_TRUE(context.CmdPresent(window));
  image = {};
  EXPECT_THROW(std::move(context).Submit(), std::bad_alloc);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 0);
    gate->complete = true;
  }
  gate->changed.notify_all();
  std::unique_lock<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->destroyed == 1; }));
}

TEST(RhiCommands, DeviceShutdownDrainsSubmittedResources) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto resource = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto *native = new TestCommandContext(queue.Get(), gate);
  CommandContext context = native;
  ASSERT_TRUE(native->CmdUseResource(resource.Get()));
  resource.Reset();
  auto token = std::move(context).Submit();
  ASSERT_TRUE(token);
  token.Release();
  queue.Release();
  EXPECT_TRUE(device->IsUniquelyOwned());
  EXPECT_EQ(gate->resources_destroyed, 0);
  // TestDevice makes the simulated GPU complete only when shutdown starts.
  device.Reset();
  EXPECT_EQ(gate->resources_destroyed, 1);
  EXPECT_EQ(gate->devices_destroyed, 1);
}

TEST(RhiCommands, CallerReleasesRecordingBeforeWindowAndDevice) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto window = ObjectPtr<TestWindow>::Adopt(new TestWindow(gate));
  auto image = Image(new TestImage(gate, window.Get()));
  auto context = CommandContext(new TestCommandContext(queue.Get(), gate));
  ASSERT_TRUE(context.CmdClearImage(image, {0.0f, 0.0f, 0.0f, 1.0f}));
  image.Release();
  EXPECT_TRUE(window->IsUniquelyOwned());
  EXPECT_EQ(gate->destroyed, 0);
  context.Release();
  EXPECT_EQ(gate->destroyed, 1);
  EXPECT_EQ(gate->windows_destroyed, 0);
  window.Reset();
  EXPECT_EQ(gate->windows_destroyed, 1);
  EXPECT_EQ(gate->devices_destroyed, 0);
  device.Reset();
  EXPECT_EQ(gate->devices_destroyed, 1);
}

}  // namespace
}  // namespace grassland::rhix
