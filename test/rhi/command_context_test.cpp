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
  int devices_destroyed{};
  int windows_destroyed{};
};

class TestWindow final : public backend::Window {
 public:
  explicit TestWindow(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestWindow() override {
    EXPECT_FALSE(HasFrameLeases());
    EXPECT_EQ(gate_->devices_destroyed, 0);
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->windows_destroyed;
  }

  ObjectPtr<backend::Image> AcquireNativeFrame() override {
    return {};
  }

  bool SetNativePresentMode(PresentMode) override {
    return true;
  }

  PresentMode GetPresentMode() const override {
    return PresentMode::VSync;
  }

  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override {
    return {{PresentMode::VSync, ImageFormat::RGBA8Unorm}};
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestImage final : public backend::Image {
 public:
  explicit TestImage(std::shared_ptr<Gate> gate, backend::Window *window = nullptr)
      : backend::Image({16, 16, ImageFormat::RGBA8Unorm}),
        gate_(std::move(gate)),
        window_(window) {
    if (window_)
      window_->AcquireFrameLease();
  }

  ~TestImage() override {
    if (window_)
      window_->ReleaseFrameLease();
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

class TestCommandContext final : public backend::CommandContext {
 public:
  TestCommandContext(std::shared_ptr<Gate> gate, QueueType type, bool fail_after_submit = false)
      : backend::CommandContext(gate.get(), type),
        gate_(std::move(gate)),
        fail_after_submit_(fail_after_submit) {
  }

  SubmitResult SubmitNative() override {
    if (fail_after_submit_) {
      MarkSubmitted();
      throw std::bad_alloc{};
    }
    return SubmitResult::Submitted;
  }

  SubmissionStatus WaitForProgress() override {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->changed.wait_for(lock, std::chrono::milliseconds(10), [&] { return gate_->complete; });
    return gate_->complete ? SubmissionStatus::Completed : SubmissionStatus::Pending;
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

  std::unique_ptr<backend::CommandContext> CreateCommandContext() const override {
    return {};
  }
};

TEST(RhiCommands, RecordingAndSubmissionKeepExactImageWithoutKeepingPayloadInToken) {
  static_assert(!std::is_copy_constructible_v<CommandContext>);
  static_assert(!std::is_copy_constructible_v<RecordedCommands>);
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QueueType::Graphics));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.ClearColor(image, {0, 0, 1, 1}));
  ASSERT_TRUE(context.Present(image));
  image = {};
  auto recorded = context.Finish();
  ASSERT_FALSE(context);
  ASSERT_TRUE(recorded);
  auto token = queue.Submit(std::move(recorded));
  ASSERT_FALSE(recorded);
  ASSERT_TRUE(token);
  EXPECT_EQ(token.GetStatus(), SubmissionStatus::Pending);
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
  device.Reset();
  EXPECT_EQ(gate->devices_destroyed, 1);
  EXPECT_FALSE(token);
}

TEST(RhiCommands, DroppedTokenStillRetiresSingleSubmissionAutomatically) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QueueType::Graphics));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  ASSERT_TRUE(context.Present(image));
  ASSERT_TRUE(queue.Submit(context.Finish()));
  image = {};
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->complete = true;
  }
  gate->changed.notify_all();
  std::unique_lock<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->destroyed == 1; }));
}

TEST(RhiCommands, CancellationReleasesClaimAndForeignQueueConsumesRejectedPacket) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto image = Image(new TestImage(gate));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  context = {};
  context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  ASSERT_TRUE(context.Present(image));
  auto packet = context.Finish();
  auto other = ObjectPtr<TestDevice>::Adopt(new TestDevice(std::make_shared<Gate>()));
  auto queue = Queue(new TestQueue(other, QueueType::Graphics));
  EXPECT_FALSE(queue.Submit(std::move(packet)));
  EXPECT_FALSE(packet);
  image = {};
  EXPECT_EQ(gate->destroyed, 1);
}

TEST(RhiCommands, InvalidAndIncompleteContextsCannotBeSubmitted) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto image = Image(new TestImage(gate));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Transfer));
  EXPECT_FALSE(context.ClearColor(image, {1, 0, 0, 1}));
  EXPECT_FALSE(context.Finish());
  context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  EXPECT_FALSE(context.Finish());
  EXPECT_FALSE(context);
}

TEST(RhiCommands, AliasedQueueDependencyDoesNotWaitForGpuOnCpu) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto transfer = Queue(new TestQueue(device, QueueType::Transfer));
  auto graphics = Queue(new TestQueue(device, QueueType::Graphics));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Transfer));
  auto upload = transfer.Submit(context.Finish());
  ASSERT_TRUE(upload);
  context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  auto done = graphics.Submit(context.Finish(), {{upload}});
  ASSERT_TRUE(done);
  EXPECT_EQ(done.GetStatus(), SubmissionStatus::Pending);
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
  auto queue = Queue(new TestQueue(device, QueueType::Graphics));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics, true));
  auto image = Image(new TestImage(gate));
  ASSERT_TRUE(context.ClearColor(image, {0, 0, 1, 1}));
  ASSERT_TRUE(context.Present(image));
  image = {};
  EXPECT_FALSE(queue.Submit(context.Finish()));
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 0);
    gate->complete = true;
  }
  gate->changed.notify_all();
  std::unique_lock<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->destroyed == 1; }));
}

TEST(RhiCommands, DeviceShutdownDrainsWorkAndReclaimsWindowAfterFrameLease) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(new TestQueue(device, QueueType::Graphics));
  auto managed = device->ManageWindow(std::make_unique<TestWindow>(gate));
  auto *window_backend = managed.Get();
  auto window = Window(ObjPtr<rhi::IWindow>::Retain((managed).Get()).Detach());
  managed.Reset();
  auto image = Image(new TestImage(gate, window_backend));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {0, 0, 0, 1}));
  ASSERT_TRUE(context.Present(image));
  image.Release();
  window.Release();
  ASSERT_TRUE(window_backend->IsUniquelyOwned());
  ASSERT_TRUE(window_backend->HasFrameLeases());
  auto token = queue.Submit(context.Finish());
  ASSERT_TRUE(token);
  token.Release();
  queue.Release();
  EXPECT_TRUE(device->IsUniquelyOwned());
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 0);
    EXPECT_EQ(gate->windows_destroyed, 0);
  }
  // TestDevice makes the simulated GPU complete only when shutdown starts.
  device.Reset();
  EXPECT_EQ(gate->destroyed, 1);
  EXPECT_EQ(gate->windows_destroyed, 1);
  EXPECT_EQ(gate->devices_destroyed, 1);
}

TEST(RhiCommands, CancelingRecordingReturnsFrameLeaseWithoutOwningWindow) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto window = device->ManageWindow(std::make_unique<TestWindow>(gate));
  auto image = Image(new TestImage(gate, window.Get()));
  auto context = CommandContext(new TestCommandContext(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {0, 0, 0, 1}));
  image.Release();
  EXPECT_TRUE(window->HasFrameLeases());
  context.Release();
  EXPECT_FALSE(window->HasFrameLeases());
  window.Reset();
  device.Reset();
  EXPECT_EQ(gate->windows_destroyed, 1);
}

}  // namespace
}  // namespace grassland::rhix
