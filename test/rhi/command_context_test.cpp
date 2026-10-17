#include <gtest/gtest.h>

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
    return {{PRESENT_MODE_VSYNC, IMAGE_FORMAT_RGBA8_UNORM}};
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestImage final : public backend::Image {
 public:
  explicit TestImage(std::shared_ptr<Gate> gate, backend::Window *window = nullptr)
      : backend::Image({16, 16, IMAGE_FORMAT_RGBA8_UNORM}),
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

class TestCommandContext final : public backend::CommandContext {
 public:
  TestCommandContext(std::shared_ptr<Gate> gate, bool fail_after_submit = false)
      : backend::CommandContext(gate.get()),
        gate_(std::move(gate)),
        fail_after_submit_(fail_after_submit) {
  }

  std::vector<std::array<float, 4>> recorded_colors;
  int recorded_presents{};
  int ended{};

  bool CmdUseResource(rhi::IObject *resource) {
    return RetainResource(resource);
  }

  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override {
#if !defined(NDEBUG)
    if (!image) {
      recording_failed_ = true;
      return false;
    }
#endif
    if (!RetainResource(image))
      return false;
    recorded_colors.push_back(color);
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

  SubmitResult SubmitNative() override {
    EXPECT_EQ(ended, 1);
    if (fail_after_submit_) {
      MarkSubmitted();
      throw std::bad_alloc{};
    }
    return SUBMIT_RESULT_SUBMITTED;
  }

  SubmissionStatus WaitForProgress() override {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->changed.wait_for(lock, std::chrono::milliseconds(10), [&] { return gate_->complete; });
    return gate_->complete ? SUBMISSION_STATUS_COMPLETED : SUBMISSION_STATUS_PENDING;
  }

 private:
  std::shared_ptr<Gate> gate_;
  bool fail_after_submit_;
};

class TestDevice final : public backend::Device {
 public:
  explicit TestDevice(std::shared_ptr<Gate> gate) : backend::Device({}, {}), gate_(std::move(gate)) {
  }

  ~TestDevice() override {
    {
      std::lock_guard<std::mutex> lock(gate_->mutex);
      gate_->complete = true;
    }
    gate_->changed.notify_all();
    ShutdownQueues();
    ++gate_->devices_destroyed;
  }

  const void *NativeIdentity() const override {
    return gate_.get();
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestQueue final : public backend::Queue {
 public:
  TestQueue(const ObjectPtr<TestDevice> &device, QueueType type) : backend::Queue(device.Get(), type) {
  }
};

TEST(RhiCommands, CommandsAreEncodedImmediatelyAndKeepParameterValues) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  auto *native = new TestCommandContext(gate);
  CommandContext context = native;
  std::array<float, 4> color{1, 0, 0, 1};
  ASSERT_TRUE(context.CmdClearColor(image, color));
  ASSERT_EQ(native->recorded_colors.size(), 1u);
  color = {0, 1, 0, 1};
  ASSERT_TRUE(context.CmdClearColor(image, color));
  ASSERT_EQ(native->recorded_colors.size(), 2u);
  EXPECT_EQ(native->recorded_colors[0], (std::array<float, 4>{1, 0, 0, 1}));
  EXPECT_EQ(native->recorded_colors[1], color);
  EXPECT_EQ(native->ended, 0);
  image.Release();
  EXPECT_EQ(gate->destroyed, 0);
  ASSERT_TRUE(context.CmdPresent(window));
  EXPECT_EQ(native->recorded_presents, 1);
  auto token = queue.Submit(std::move(context));
  ASSERT_TRUE(token);
  EXPECT_EQ(native->ended, 1);
  EXPECT_EQ(native->recorded_colors.size(), 2u);
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
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto window = Window(new TestWindow(gate));
  auto image = ObjectPtr<TestImage>::Adopt(new TestImage(gate));
  auto resource = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto another = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto *native = new TestCommandContext(gate);
  CommandContext context = native;
  ASSERT_TRUE(native->CmdUseResource(resource.Get()));
  ASSERT_TRUE(native->CmdClearColor(image.Get(), {0, 0, 1, 1}));
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
  auto token = queue.Submit(std::move(context));
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
  auto context = ObjectPtr<TestCommandContext>::Adopt(new TestCommandContext(gate));
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
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto context = CommandContext(new TestCommandContext(gate));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {0, 0, 1, 1}));
  ASSERT_TRUE(context.CmdPresent(window));
  image = {};
  auto recorded = std::move(context);
  ASSERT_FALSE(context);
  ASSERT_TRUE(recorded);
  auto token = queue.Submit(std::move(recorded));
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

TEST(RhiCommands, DroppedTokenStillRetiresSingleSubmissionAutomatically) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto context = CommandContext(new TestCommandContext(gate));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {1, 0, 0, 1}));
  ASSERT_TRUE(context.CmdPresent(window));
  ASSERT_TRUE(queue.Submit(std::move(context)));
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
TEST(RhiCommands, CancellationReleasesResourcesAndForeignQueueConsumesRejectedPacket) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  auto context = CommandContext(new TestCommandContext(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {1, 0, 0, 1}));
  context = {};
  context = CommandContext(new TestCommandContext(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {1, 0, 0, 1}));
  ASSERT_TRUE(context.CmdPresent(window));
  auto packet = std::move(context);
  auto other = ObjectPtr<TestDevice>::Adopt(new TestDevice(std::make_shared<Gate>()));
  auto queue = Queue(new TestQueue(other, QUEUE_TYPE_GRAPHICS));
  EXPECT_FALSE(queue.Submit(std::move(packet)));
  EXPECT_FALSE(packet);
  image = {};
  EXPECT_EQ(gate->destroyed, 1);
}

TEST(RhiCommands, SealedContextsCannotRecordOrSubmitAgain) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto image = Image(new TestImage(gate));
  auto native = ObjectPtr<TestCommandContext>::Adopt(new TestCommandContext(gate));
  auto context = CommandContext(ObjPtr<rhi::ICommandContext>::Retain(native.Get()).Detach());
  ASSERT_TRUE(context.CmdClearColor(image, {1, 0, 0, 1}));
  auto token = queue.Submit(std::move(context));
  ASSERT_TRUE(token);
  EXPECT_FALSE(native->CmdClearColor(image.Get(), {0, 1, 0, 1}));
  EXPECT_EQ(queue.Get()->Submit(native.Get()), nullptr);
  native.Reset();
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  EXPECT_TRUE(token.WaitHost());
}

#endif

TEST(RhiCommands, AliasedQueueDependencyDoesNotWaitForGpuOnCpu) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto transfer = Queue(new TestQueue(device, QUEUE_TYPE_TRANSFER));
  auto graphics = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto context = CommandContext(new TestCommandContext(gate));
  auto upload = transfer.Submit(std::move(context));
  ASSERT_TRUE(upload);
  context = CommandContext(new TestCommandContext(gate));
  auto done = graphics.Submit(std::move(context), {{upload}});
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
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto context = CommandContext(new TestCommandContext(gate, true));
  auto window = Window(new TestWindow(gate));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {0, 0, 1, 1}));
  ASSERT_TRUE(context.CmdPresent(window));
  image = {};
  EXPECT_FALSE(queue.Submit(std::move(context)));
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
  auto queue = Queue(new TestQueue(device, QUEUE_TYPE_GRAPHICS));
  auto resource = ObjectPtr<TestResource>::Adopt(new TestResource(gate));
  auto *native = new TestCommandContext(gate);
  CommandContext context = native;
  ASSERT_TRUE(native->CmdUseResource(resource.Get()));
  resource.Reset();
  auto token = queue.Submit(std::move(context));
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
  auto window = ObjectPtr<TestWindow>::Adopt(new TestWindow(gate));
  auto image = Image(new TestImage(gate, window.Get()));
  auto context = CommandContext(new TestCommandContext(gate));
  ASSERT_TRUE(context.CmdClearColor(image, {0, 0, 0, 1}));
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
