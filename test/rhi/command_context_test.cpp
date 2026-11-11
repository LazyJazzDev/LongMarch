#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <type_traits>

#include "grassland/rhi/backend/backend_command_context_state.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_queue_state.h"
#include "grassland/rhi/backend/backend_timeline_state.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_state.h"
#include "grassland/rhi/rhix/rhix.h"

using grassland::rhi::backend::MatchInterfaces;

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;

namespace {
using backend::ObjectPtr;

struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool notified{};
  unsigned completed_polls{};
  bool pause_preparation{};
  bool preparation_entered{};
  bool complete{};
  bool reject_native{};
  uint64_t completed_limit{UINT64_MAX};
  int destroyed{};
  int resources_destroyed{};
  int devices_destroyed{};
  int windows_destroyed{};
};

class TestWindow final : public backend::Object<backend::IBackendWindow> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IWindow, backend::IBackendWindow, rhi::IWindow)

 public:
  explicit TestWindow(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestWindow() override {
    EXPECT_EQ(gate_->devices_destroyed, 0);
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->windows_destroyed;
    DestroyWindow();
  }

  ObjectPtr<backend::IBackendImage> AcquireNativeFrame() override {
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

 private:
  backend::WindowState state_;

  backend::WindowState &StateWindow() override {
    return state_;
  }

  const backend::WindowState &StateWindow() const override {
    return state_;
  }

 public:
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override {
    return backend::IBackendWindow::CommonEnableImGui(enabled, font_file_path, font_size);
  }

  bool UpdateImGui(const std::function<void()> &callback) override {
    return backend::IBackendWindow::CommonUpdateImGui(callback);
  }

  EventManager<void(int, int)> &ResizeEvent() override {
    return backend::IBackendWindow::CommonResizeEvent();
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return backend::IBackendWindow::CommonFramebufferResizeEvent();
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return backend::IBackendWindow::CommonCursorEnterEvent();
  }

  EventManager<void(bool)> &FocusEvent() override {
    return backend::IBackendWindow::CommonFocusEvent();
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return backend::IBackendWindow::CommonMouseMoveEvent();
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return backend::IBackendWindow::CommonMouseButtonEvent();
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return backend::IBackendWindow::CommonScrollEvent();
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return backend::IBackendWindow::CommonKeyEvent();
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return backend::IBackendWindow::CommonCharEvent();
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return backend::IBackendWindow::CommonDropEvent();
  }

  rhi::IImage *GetNextImage() override {
    return backend::IBackendWindow::CommonGetNextImage();
  }

  bool IsSwapChainInvalid() const override {
    return backend::IBackendWindow::CommonIsSwapChainInvalid();
  }

  bool RebuildSwapChain() override {
    return backend::IBackendWindow::CommonRebuildSwapChain();
  }

  bool SetPresentMode(PresentMode mode) override {
    return backend::IBackendWindow::CommonSetPresentMode(mode);
  }

  bool SetColorSpace(ColorSpace color_space) override {
    return backend::IBackendWindow::CommonSetColorSpace(color_space);
  }

  ColorSpace GetColorSpace() const override {
    return backend::IBackendWindow::CommonGetColorSpace();
  }

  bool GetHDRReferenceWhiteNits(float &nits) const override {
    return backend::IBackendWindow::CommonGetHDRReferenceWhiteNits(nits);
  }

  float GetHDRReferenceWhiteScale() const override {
    return backend::IBackendWindow::CommonGetHDRReferenceWhiteScale();
  }

  float GetHDRHeadroom() const override {
    return backend::IBackendWindow::CommonGetHDRHeadroom();
  }

  GLFWwindow *GLFWWindow() const override {
    return backend::IBackendWindow::CommonGLFWWindow();
  }

  bool ShouldClose() const override {
    return backend::IBackendWindow::CommonShouldClose();
  }

  bool IsKeyDown(int key) const override {
    return backend::IBackendWindow::CommonIsKeyDown(key);
  }

  bool SetTitle(const std::string &title) override {
    return backend::IBackendWindow::CommonSetTitle(title);
  }
};

class TestImage final : public backend::Object<backend::IBackendImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, backend::IBackendImage, rhi::IImage)

 public:
  explicit TestImage(std::shared_ptr<Gate> gate, backend::IBackendWindow *window = nullptr)
      : gate_(std::move(gate)),
        window_(window) {
    InitializeImage({16, 16, IMAGE_FORMAT_R8G8B8A8_UNORM});
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
  backend::IBackendWindow *window_;

 private:
  backend::ImageState state_;

  backend::ImageState &StateImage() override {
    return state_;
  }

  const backend::ImageState &StateImage() const override {
    return state_;
  }

 public:
  const ImageDesc &GetDesc() const override {
    return backend::IBackendImage::CommonGetDesc();
  }

  bool SupportsStorage() const override {
    return backend::IBackendImage::CommonSupportsStorage();
  }
};

class TestResource final : public backend::Object<rhi::IObject> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IObject, rhi::IObject)

  explicit TestResource(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
  }

  ~TestResource() override {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->resources_destroyed;
  }

 private:
  std::shared_ptr<Gate> gate_;
};

class TestTimeline final : public backend::Object<backend::IBackendTimeline> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(ITimeline, backend::IBackendTimeline, ITimeline)

 public:
  TestTimeline(backend::IBackendDevice *device, std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
    InitializeTimeline(device);
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
    gate_->notified = true;
    gate_->changed.notify_all();
  }

 private:
  uint64_t value_{};
  std::shared_ptr<Gate> gate_;

 private:
  backend::TimelineState state_;

  backend::TimelineState &StateTimeline() override {
    return state_;
  }

  const backend::TimelineState &StateTimeline() const override {
    return state_;
  }
};

class TestCommandContext final : public backend::Object<backend::IBackendCommandContext> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICommandContext, backend::IBackendCommandContext, rhi::ICommandContext)

 public:
  TestCommandContext(rhi::IQueue *queue, std::shared_ptr<Gate> gate, bool fail_after_submit = false)
      : gate_(std::move(gate)),
        fail_after_submit_(fail_after_submit) {
    InitializeCommandContext(static_cast<backend::IBackendQueue *>(queue));
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
      StateCommandContext().recording_failed_ = true;
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
    if (StateCommandContext().sealed_ || !window)
      return false;
#endif
    if (StateCommandContext().recording_failed_)
      return false;
    ++recorded_presents;
    return true;
  }

  bool EndRecording() override {
    ++ended;
    return true;
  }

  void UpdatePresentedImages() override {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->preparation_entered = true;
    gate_->changed.notify_all();
    gate_->changed.wait(lock, [&] { return !gate_->pause_preparation; });
  }

  SubmitResult SubmitNative(const SubmitInfo &, Event event) override {
    if (gate_->reject_native)
      return SUBMIT_RESULT_REJECTED;
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
    if (completed < signal.value)
      return SUBMISSION_STATUS_PENDING;
    std::lock_guard<std::mutex> lock(gate_->mutex);
    ++gate_->completed_polls;
    gate_->changed.notify_all();
    return SUBMISSION_STATUS_COMPLETED;
  }

 private:
  std::shared_ptr<Gate> gate_;
  bool fail_after_submit_;

 private:
  backend::CommandContextState state_;

  backend::CommandContextState &StateCommandContext() override {
    return state_;
  }

  const backend::CommandContextState &StateCommandContext() const override {
    return state_;
  }

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override {
    return backend::IBackendCommandContext::CommonCmdDrawImGui(window, target);
  }

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override {
    return backend::IBackendCommandContext::CommonCmdPresentImage(window, frame_img, frame_color_space);
  }

  bool CmdQueryCompactedSize(IAccelerationStructure *structure) override {
    return backend::IBackendCommandContext::CommonCmdQueryCompactedSize(structure);
  }

  bool CmdCompactAccelerationStructure(IAccelerationStructure *source, IAccelerationStructure *destination) override {
    return backend::IBackendCommandContext::CommonCmdCompactAccelerationStructure(source, destination);
  }

  bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<rhi::AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override {
    return backend::IBackendCommandContext::CommonCmdBuildBottomLevelAccelerationStructure(destination, geometries,
                                                                                           mode);
  }

  bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<rhi::RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override {
    return backend::IBackendCommandContext::CommonCmdBuildTopLevelAccelerationStructure(destination, instances, mode);
  }

  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override {
    return backend::IBackendCommandContext::CommonCmdBindRayTracingProgram(program);
  }

  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override {
    return backend::IBackendCommandContext::CommonCmdDispatchRays(width, height, depth, sbt);
  }

  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override {
    return backend::IBackendCommandContext::CommonSubmit(info);
  }

  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override {
    return backend::IBackendCommandContext::CommonCmdUploadBuffer(buffer, offset, data, size);
  }

  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override {
    return backend::IBackendCommandContext::CommonCmdCopyBuffer(source, source_offset, destination, destination_offset,
                                                                size);
  }

  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override {
    return backend::IBackendCommandContext::CommonCmdBindGroup(index, group);
  }

  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override {
    return backend::IBackendCommandContext::CommonCmdUploadImage(image, data, row_pitch);
  }

  bool CmdBindComputeProgram(IComputeProgram *program) override {
    return backend::IBackendCommandContext::CommonCmdBindComputeProgram(program);
  }

  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override {
    return backend::IBackendCommandContext::CommonCmdDispatch(x, y, z);
  }

  bool CmdBindProgram(IProgram *program) override {
    return backend::IBackendCommandContext::CommonCmdBindProgram(program);
  }

  bool CmdBeginRendering(const rhi::RenderingInfo &info) override {
    return backend::IBackendCommandContext::CommonCmdBeginRendering(info);
  }

  bool CmdEndRendering() override {
    return backend::IBackendCommandContext::CommonCmdEndRendering();
  }

  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) override {
    return backend::IBackendCommandContext::CommonCmdBindVertexBuffer(binding, buffer, offset);
  }

  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) override {
    return backend::IBackendCommandContext::CommonCmdBindIndexBuffer(buffer, format, offset);
  }

  bool CmdSetViewport(const Viewport &viewport) override {
    return backend::IBackendCommandContext::CommonCmdSetViewport(viewport);
  }

  bool CmdSetScissor(const Scissor &scissor) override {
    return backend::IBackendCommandContext::CommonCmdSetScissor(scissor);
  }

  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0) override {
    return backend::IBackendCommandContext::CommonCmdDraw(vertex_count, instance_count, first_vertex, first_instance);
  }

  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override {
    return backend::IBackendCommandContext::CommonCmdDrawIndexed(index_count, instance_count, first_index,
                                                                 vertex_offset, first_instance);
  }

  IDownloadToken *CmdDownloadBuffer(IBuffer *argument0, uint64_t argument1, uint64_t argument2) override {
    return backend::IBackendCommandContext::CommonCmdDownloadBuffer(argument0, argument1, argument2);
  }

  IDownloadToken *CmdDownloadImage(IImage *argument0, const ImageDownloadRegion &argument1) override {
    return backend::IBackendCommandContext::CommonCmdDownloadImage(argument0, argument1);
  }
};

class TestQueue final : public backend::Object<backend::IBackendQueue> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IQueue, backend::IBackendQueue, rhi::IQueue)

 public:
  TestTimeline *GetTimeline() const override {
    return timeline_.get();
  }

  rhi::ICommandContext *CreateCommandContext() override {
    return new TestCommandContext(this, gate_);
  }

  TestQueue(backend::IBackendDevice *device, QueueType type, std::shared_ptr<Gate> gate)
      : gate_(std::move(gate)),
        timeline_(std::make_unique<TestTimeline>(device, gate_)) {
    InitializeQueue(device, type);
  }

 private:
  std::shared_ptr<Gate> gate_;
  std::unique_ptr<TestTimeline> timeline_;

 private:
  backend::QueueState state_;

  backend::QueueState &StateQueue() override {
    return state_;
  }

  const backend::QueueState &StateQueue() const override {
    return state_;
  }

 public:
  QueueType GetType() const override {
    return backend::IBackendQueue::CommonGetType();
  }
};

class TestDevice final : public backend::Object<backend::IBackendDevice> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, backend::IBackendDevice, rhi::IDevice)

 public:
  explicit TestDevice(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
    InitializeDevice({}, {});

    for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST})
      StateDevice().queues_[type] = ObjectPtr<backend::IBackendQueue>::Adopt(new TestQueue(this, type, gate_));
    StartCollector();
  }

  ~TestDevice() override {
    {
      std::lock_guard<std::mutex> lock(gate_->mutex);
      gate_->complete = true;
      gate_->notified = true;
      gate_->completed_limit = UINT64_MAX;
    }
    gate_->changed.notify_all();
    ShutdownQueues();
    ++gate_->devices_destroyed;
    DestroyDevice();
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

  void InterruptCollector() override {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    gate_->notified = true;
    gate_->changed.notify_all();
  }

  void WaitForProgress(const std::vector<Event> &next) override {
    for (auto event : next) {
      uint64_t value;
      if (event.timeline->GetCompletedValue(value) && value >= event.value)
        return;
    }
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->changed.notify_all();
    gate_->changed.wait(lock, [&] { return gate_->notified; });
    gate_->notified = false;
  }

  std::shared_ptr<Gate> gate_;

 private:
  backend::DeviceState state_;

  backend::DeviceState &StateDevice() override {
    return state_;
  }

  const backend::DeviceState &StateDevice() const override {
    return state_;
  }

 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override {
    return backend::IBackendDevice::CommonCreateBottomLevelAccelerationStructure(desc);
  }

  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override {
    return backend::IBackendDevice::CommonCreateTopLevelAccelerationStructure(desc);
  }

  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override {
    return backend::IBackendDevice::CommonCreateRayTracingProgram(desc);
  }

  std::string Name() const override {
    return backend::IBackendDevice::CommonName();
  }

  bool SupportRayTracingPipeline() const override {
    return backend::IBackendDevice::CommonSupportRayTracingPipeline();
  }

  bool SupportRayQuery() const override {
    return backend::IBackendDevice::CommonSupportRayQuery();
  }

  bool SupportGeometryShader() const override {
    return backend::IBackendDevice::CommonSupportGeometryShader();
  }

  rhi::IImage *CreateImage(const ImageDesc &desc) override {
    return backend::IBackendDevice::CommonCreateImage(desc);
  }

  ISampler *CreateSampler(const SamplerDesc &desc = {}) override {
    return backend::IBackendDevice::CommonCreateSampler(desc);
  }

  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override {
    return backend::IBackendDevice::CommonCreateBindingGroup(layout);
  }

  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override {
    return backend::IBackendDevice::CommonCreateBuffer(desc);
  }

  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override {
    return backend::IBackendDevice::CommonCreateComputeProgram(desc);
  }

  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override {
    return backend::IBackendDevice::CommonCreateProgram(desc);
  }

  rhi::IShader *CreateShader(const ShaderDesc &desc) override {
    return backend::IBackendDevice::CommonCreateShader(desc);
  }

  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override {
    return backend::IBackendDevice::CommonCreateGLFWWindow(desc);
  }

  rhi::IQueue *GetQueue(QueueType type) override {
    return backend::IBackendDevice::CommonGetQueue(type);
  }

  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override {
    return backend::IBackendDevice::CommonWaitEvents(events, wait_all);
  }
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
    gate->notified = true;
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
    gate->notified = true;
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
    gate->notified = true;
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

TEST(RhiCommands, SubmissionCanRetireWhileNextFramePreparationIsBlocked) {
  auto gate = std::make_shared<Gate>();
  gate->complete = true;
  gate->pause_preparation = true;
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  auto submitted = std::async(std::launch::async, [&] {
    auto context = queue.CreateCommandContext();
    return std::move(context).Submit();
  });
  bool collected;
  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    collected = gate->changed.wait_for(lock, std::chrono::seconds(2),
                                       [&] { return gate->preparation_entered && gate->completed_polls == 1; });
    gate->pause_preparation = false;
  }
  gate->changed.notify_all();
  auto token = submitted.get();
  ASSERT_TRUE(token);
  EXPECT_TRUE(collected);
  EXPECT_TRUE(token.WaitHost());
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
    gate->notified = true;
  }
  gate->changed.notify_all();
  ASSERT_TRUE(tokens[1].WaitHost());
  EXPECT_EQ(tokens[0].GetStatus(), SUBMISSION_STATUS_COMPLETED);
  EXPECT_EQ(tokens[2].GetStatus(), SUBMISSION_STATUS_PENDING);
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 2);
    gate->completed_limit = 3;
    gate->notified = true;
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
    gate->notified = true;
  }
  gate->changed.notify_all();
  std::unique_lock<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate->destroyed == 1; }));
}

#if !defined(NDEBUG)
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
    gate->notified = true;
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
    gate->notified = true;
  }
  gate->changed.notify_all();
  EXPECT_TRUE(done.WaitHost());
}

TEST(RhiCommands, RejectedNativeSubmissionRemovesItsRegisteredRecord) {
  auto gate = std::make_shared<Gate>();
  gate->reject_native = true;
  auto device = ObjectPtr<TestDevice>::Adopt(new TestDevice(gate));
  auto queue = Queue(device->GetQueue(QUEUE_TYPE_MAIN));
  {
    auto context = queue.CreateCommandContext();
    auto image = Image(new TestImage(gate));
    ASSERT_TRUE(context.CmdClearImage(image, {}));
    image.Release();
    EXPECT_FALSE(std::move(context).Submit());
  }
  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    EXPECT_EQ(gate->destroyed, 1);
    EXPECT_EQ(gate->completed_polls, 0);
    gate->reject_native = false;
    gate->complete = true;
  }
  auto context = queue.CreateCommandContext();
  auto token = std::move(context).Submit();
  ASSERT_TRUE(token);
  EXPECT_EQ(token.GetEvent().value, 2);
  EXPECT_TRUE(token.WaitHost());
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
    gate->notified = true;
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
