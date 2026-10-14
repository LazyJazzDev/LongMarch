#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <type_traits>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi.h"

namespace grassland::rhi {
namespace {
using namespace backend;

struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool complete{};
  int destroyed{};
  int devices_destroyed{};
  int windows_destroyed{};
};

class TestWindow final : public IWindow {
 public:
  explicit TestWindow(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestWindow() override {
    EXPECT_FALSE(HasFrameLeases());
    EXPECT_EQ(gate_->devices_destroyed, 0);
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->windows_destroyed;
  }

  ObjectPtr<IImage> AcquireFrame() override {
    return {};
  }

  bool SetPresentMode(PresentMode) override {
    return true;
  }

  PresentMode GetPresentMode() const override {
    return PresentMode::VSync;
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestImage final : public IImage {
 public:
  explicit TestImage(std::shared_ptr<Gate> gate, IWindow *window = nullptr)
      : IImage({16, 16, ImageFormat::RGBA8Unorm}),
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
  IWindow *window_;
};

class TestCommandContext final : public ICommandContext {
 public:
  TestCommandContext(std::shared_ptr<Gate> gate, QueueType type, bool fail_after_submit = false)
      : ICommandContext(gate.get(), type),
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

class TestDevice final : public IDevice {
 public:
  explicit TestDevice(std::shared_ptr<Gate> gate) : IDevice({}, {}), gate_(std::move(gate)) {
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

class TestQueue final : public IQueue {
 public:
  TestQueue(const ObjectPtr<TestDevice> &device, QueueType type) : IQueue(device.Get(), type) {
  }

  std::unique_ptr<ICommandContext> CreateCommandContext() const override {
    return {};
  }
};

TEST(RhiCommands, RecordingAndSubmissionKeepExactImageWithoutKeepingPayloadInToken) {
  static_assert(!std::is_copy_constructible_v<CommandContext>);
  static_assert(!std::is_copy_constructible_v<RecordedCommands>);
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Graphics));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate));
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
  auto queue = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Graphics));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate));
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
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  context = {};
  context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  ASSERT_TRUE(context.Present(image));
  auto packet = context.Finish();
  auto other = ObjectPtr<TestDevice>::Adopt(new TestDevice(std::make_shared<Gate>()));
  auto queue = Access::Adopt<Queue>(std::make_unique<TestQueue>(other, QueueType::Graphics));
  EXPECT_FALSE(queue.Submit(std::move(packet)));
  EXPECT_FALSE(packet);
  image = {};
  EXPECT_EQ(gate->destroyed, 1);
}

TEST(RhiCommands, InvalidAndIncompleteContextsCannotBeSubmitted) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Transfer));
  EXPECT_FALSE(context.ClearColor(image, {1, 0, 0, 1}));
  EXPECT_FALSE(context.Finish());
  context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
  ASSERT_TRUE(context.ClearColor(image, {1, 0, 0, 1}));
  EXPECT_FALSE(context.Finish());
  EXPECT_FALSE(context);
}

TEST(RhiCommands, AliasedQueueDependencyDoesNotWaitForGpuOnCpu) {
  auto gate = std::make_shared<Gate>();
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto transfer = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Transfer));
  auto graphics = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Graphics));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Transfer));
  auto upload = transfer.Submit(context.Finish());
  ASSERT_TRUE(upload);
  context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
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
  auto queue = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Graphics));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics, true));
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate));
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
  auto queue = Access::Adopt<Queue>(std::make_unique<TestQueue>(device, QueueType::Graphics));
  auto managed = device->ManageWindow(std::make_unique<TestWindow>(gate));
  auto *window_backend = managed.Get();
  auto window = Access::Share<Window>(managed);
  managed.Reset();
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate, window_backend));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
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
  auto image = Access::Adopt<Image>(std::make_unique<TestImage>(gate, window.Get()));
  auto context = Access::Adopt<CommandContext>(std::make_unique<TestCommandContext>(gate, QueueType::Graphics));
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
}  // namespace grassland::rhi
