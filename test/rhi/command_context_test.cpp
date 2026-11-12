#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image_presenter.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <system_error>
#include <type_traits>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_shader_impl.h"
#include "grassland/rhi/rhi_program.h"
#include "grassland/rhi/rhix/rhix.h"

namespace grassland::rhix {
using namespace rhi;
namespace backend = rhi::backend;

namespace {
using namespace backend;

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

class TestWindow final : public backend::Object<rhi::IWindow, backend::IBackendWindow> {
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

  void InitializeWindow() {
  }

 public:
  bool EnableImGui(bool enabled, const char *font_file_path, float font_size) override {
    if (!glfw::IsWindowThread() || processing_imgui_)
      return false;
    if (!enabled) {
      imgui_.reset();
      return true;
    }
    if (!imgui_)
      imgui_ =
          WindowImGui::Create(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get(), window_, font_file_path, font_size);
    return bool(imgui_);
  }

  ImGuiContext *GetImGuiContext() const override {
    return imgui_ ? imgui_->Context() : nullptr;
  }

  bool UpdateImGui(const std::function<void()> &callback) override {
    if (!imgui_)
      return true;
    if (!glfw::IsWindowThread() || processing_imgui_)
      return false;

    struct ProcessingScope {
      bool &flag;

      ProcessingScope(bool &value) : flag(value) {
        flag = true;
      }

      ~ProcessingScope() {
        flag = false;
      }
    } scope(processing_imgui_);

    return imgui_->Update(callback);
  }

  const void *DeviceIdentity() const override {
    return device_;
  }

  bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) override {
    ObjPtr<IImage> frame = GetNextImage();
    if (!frame)
      return false;
    if (!presenter_)
      presenter_ = std::make_unique<ImagePresenter>(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get());
    return presenter_->Record(commands, image, frame.Get(), color_space, GetColorSpace(), GetHDRReferenceWhiteScale());
  }

  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) override {
    return RecordImGui(commands, image, GetColorSpace(), GetHDRReferenceWhiteScale());
  }

  bool RecordImGui(rhi::ICommandContext *commands,
                   rhi::IImage *image,
                   ColorSpace color_space,
                   float white_scale) override {
    if (!imgui_)
      return true;
    if (!glfw::IsWindowThread() || processing_imgui_)
      return false;

    struct ProcessingScope {
      bool &flag;

      ProcessingScope(bool &value) : flag(value) {
        flag = true;
      }

      ~ProcessingScope() {
        flag = false;
      }
    } scope(processing_imgui_);

    return imgui_->Record(commands, image, color_space, white_scale);
  }

  rhi::IImage *GetNextImage() override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread()) {
      ReportError(-1, "GetNextImage must run on the window thread.");
      std::abort();
    }
#endif
    if (invalid_)
      return nullptr;
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
      return nullptr;
    std::lock_guard<std::mutex> lock(next_image_mutex_);
#if !defined(NDEBUG)
    if (!next_image_) {
      ReportError(-1, "No prepared swapchain image for a drawable window.");
      std::abort();
    }
#endif
    return rhi::QueryInterfacePtr<rhi::IImage>(next_image_.Get()).Detach();
  }

  IBackendQueue *OwnerQueue() const override {
    return queue_;
  }

  void ReleaseNextImage() override {
    std::lock_guard<std::mutex> lock(next_image_mutex_);
    next_image_.Reset();
  }

  bool IsSwapChainInvalid() const override {
    return invalid_;
  }

  bool RebuildSwapChain() override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread())
      return false;
#endif
    const auto options = GetSupportedPresentationOptions();
    if (std::none_of(options.begin(), options.end(),
                     [&](const auto &option) { return option.color_space == requested_color_space_; })) {
      ReportError(-1, "Requested swapchain color space is unavailable.");
      return false;
    }
    invalid_ = true;
    ReleaseNextImage();
    if (!RebuildNativeSwapChain(requested_mode_))
      return false;
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
      return true;
    color_space_ = requested_color_space_;
    invalid_ = false;
    if (UpdateNextImage())
      return true;
    invalid_ = true;
    return false;
  }

  bool SetPresentMode(PresentMode mode) override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread())
      return false;
#endif
    requested_mode_ = mode;
    return RebuildSwapChain();
  }

  bool SetColorSpace(ColorSpace color_space) override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread())
      return false;
#endif
    const auto options = GetSupportedPresentationOptions();
    if (std::none_of(options.begin(), options.end(),
                     [&](const auto &option) { return option.color_space == color_space; }))
      return false;
    const auto previous = requested_color_space_;
    requested_color_space_ = color_space;
    if (RebuildSwapChain())
      return true;
    requested_color_space_ = previous;
    return false;
  }

  ColorSpace GetColorSpace() const override {
    return color_space_;
  }

  float GetHDRReferenceWhiteScale() const override {
#ifdef _WIN32
    float nits;
    return GetHDRReferenceWhiteNits(nits) ? nits / 80.0f : 1.0f;
#else
    // Linux uses fixed source encoding parameters; macOS uses relative EDR.
    return 1.0f;
#endif
  }

  float GetHDRHeadroom() const override {
#ifdef _WIN32
    float white;
    if (!GetHDRReferenceWhiteNits(white))
      return 10.0f;
    const auto monitor = MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST);
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
      return 10.0f;
    for (UINT a = 0;; ++a) {
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
      if (factory->EnumAdapters1(a, &adapter) != S_OK)
        break;
      for (UINT o = 0;; ++o) {
        Microsoft::WRL::ComPtr<IDXGIOutput> output;
        if (adapter->EnumOutputs(o, &output) != S_OK)
          break;
        Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
        DXGI_OUTPUT_DESC1 desc{};
        if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)) || desc.Monitor != monitor)
          continue;
        const float ratio = desc.MaxLuminance / white;
        return std::isfinite(ratio) && ratio > 0.0f ? std::max(1.0f, ratio) : 10.0f;
      }
    }
#endif
    return 10.0f;
  }

  bool GetHDRReferenceWhiteNits(float &nits) const override {
    if (!glfw::IsWindowThread())
      return false;
#ifdef _WIN32
    MONITORINFOEXW monitor{};
    monitor.cbSize = sizeof(monitor);
    if (!GetMonitorInfoW(MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST), &monitor))
      return false;
    UINT32 path_count{}, mode_count{};
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS)
      return false;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) !=
        ERROR_SUCCESS)
      return false;
    for (UINT32 i = 0; i < path_count; ++i) {
      const auto &path = paths[i];
      DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
      source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId,
                       path.sourceInfo.id};
      if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
          wcscmp(source.viewGdiDeviceName, monitor.szDevice))
        continue;
      DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
      color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color), path.targetInfo.adapterId,
                      path.targetInfo.id};
      if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS || !color.advancedColorEnabled)
        return false;
      DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
      white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(white), path.targetInfo.adapterId,
                      path.targetInfo.id};
      if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS || !white.SDRWhiteLevel)
        return false;
      nits = 80.0f * (white.SDRWhiteLevel / 1000.0f);
      return true;
    }
#endif
    return false;
  }

 private:
  ColorSpace RequestedColorSpace() const {
    return requested_color_space_;
  }

  void DestroyWindow() {
    imgui_.reset();
    glfw::UnregisterWindow(window_);
    glfw::DestroyWindow(window_);
  }

  bool InitializeWindow(IBackendDevice *device, const WindowDesc &desc) {
    device_ = device;
    queue_ = rhi::QueryInterfacePtr<IBackendQueue>(
                 ObjectPtr<IQueue>::Adopt(rhi::QueryInterfacePtr<IDevice>(device)->GetQueue(QUEUE_TYPE_MAIN)).Get())
                 .Get();
    requested_mode_ = desc.present_mode;
    requested_color_space_ = desc.color_space;
    window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
    if (!window_)
      return false;
    glfw::RegisterWindow(this, this);
    return true;
  }

 public:
  GLFWwindow *GLFWWindow() const override {
    return window_;
  }

  bool ShouldClose() const override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread())
      return true;
#endif
    return glfwWindowShouldClose(window_);
  }

  bool IsKeyDown(int key) const override {
#if !defined(NDEBUG)
    if (!glfw::IsWindowThread())
      return false;
#endif
    return glfwGetKey(window_, key) == GLFW_PRESS;
  }

  bool SetTitle(const std::string &title) override {
    return glfw::SetWindowTitle(window_, title.c_str());
  }

  EventManager<void(int, int)> &ResizeEvent() override {
    return resize_event_;
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return framebufferresize_event_;
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return cursorenter_event_;
  }

  EventManager<void(bool)> &FocusEvent() override {
    return focus_event_;
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return mousemove_event_;
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return mousebutton_event_;
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return scroll_event_;
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return key_event_;
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return char_event_;
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return drop_event_;
  }

 private:
  bool invalid_{true};
  IBackendDevice *device_{};
  IBackendQueue *queue_{};
  std::unique_ptr<WindowImGui> imgui_{};
  std::unique_ptr<ImagePresenter> presenter_{};
  bool processing_imgui_{};
  EventManager<void(int, int)> resize_event_{};
  EventManager<void(int, int)> framebufferresize_event_{};
  EventManager<void(bool)> cursorenter_event_{};
  EventManager<void(bool)> focus_event_{};
  EventManager<void(double, double)> mousemove_event_{};
  EventManager<void(int, int, int, double, double)> mousebutton_event_{};
  EventManager<void(double, double)> scroll_event_{};
  EventManager<void(int, int, int, int)> key_event_{};
  EventManager<void(uint32_t)> char_event_{};
  EventManager<void(int, const char **)> drop_event_{};
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  ColorSpace requested_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  ColorSpace color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_{};
  ObjectPtr<IBackendImage> next_image_{};
};

class TestImage final : public backend::Object<rhi::IImage, backend::IBackendImage> {
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

  ImageDesc desc_{};
  backend::IBackendQueue *queue_{};

  void InitializeImage(ImageDesc desc, backend::IBackendQueue *queue = nullptr) {
    desc_ = desc;
    queue_ = queue;
  }

 public:
  const ImageDesc &GetDesc() const override {
    return desc_;
  }

  bool SupportsStorage() const override {
    return false;
  }

  bool SupportsQueue(const backend::IBackendQueue *queue) const override {
    return !queue_ || queue_ == queue;
  }
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

class TestTimeline final : public backend::Object<rhi::ITimeline, backend::IBackendTimeline> {
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

  backend::IBackendDevice *Owner() const override {
    return owner_;
  }

 private:
  uint64_t value_{};

  std::shared_ptr<Gate> gate_;

  void InitializeTimeline(backend::IBackendDevice *device) {
    owner_ = device;
  }

  backend::IBackendDevice *owner_{};
};

class TestCommandContext final : public backend::Object<rhi::ICommandContext, backend::IBackendCommandContext> {
 public:
  TestCommandContext(rhi::IQueue *queue, std::shared_ptr<Gate> gate, bool fail_after_submit = false)
      : gate_(std::move(gate)),
        fail_after_submit_(fail_after_submit) {
    InitializeCommandContext(rhi::QueryInterfacePtr<backend::IBackendQueue>(queue).Get());
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

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override {
    auto native = rhi::QueryInterfacePtr<IBackendWindow>(window).Get();
    auto image = rhi::QueryInterfacePtr<IBackendImage>(target).Get();
    if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
        (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) ||
        !glfw::IsWindowThread() || sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
      recording_failed_ = true;
      ReportError(-1,
                  "CmdDrawImGui requires a window and target from this device outside rendering on the window thread.");
      return false;
    }
    const auto format = target->GetDesc().format;
    if (!IsColorFormat(format) || IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format)) {
      recording_failed_ = true;
      ReportError(-1, "CmdDrawImGui requires a non-integer color target.");
      return false;
    }
    if (!RetainResource(target))
      return false;
    if (!native->RecordImGui(this, target, COLOR_SPACE_BT709_LINEAR, 1.0f)) {
      recording_failed_ = true;
      return false;
    }
    return true;
  }

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override {
    auto native = rhi::QueryInterfacePtr<IBackendWindow>(window).Get();
    auto image = rhi::QueryInterfacePtr<IBackendImage>(frame_img).Get();
    if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
        (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) ||
        !glfw::IsWindowThread() || sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
      recording_failed_ = true;
      ReportError(
          -1, "CmdPresentImage requires a window and image from this device outside rendering on the window thread.");
      return false;
    }
    if (!RetainResource(frame_img))
      return false;
    if (!native->RecordPresentImage(this, frame_img, frame_color_space)) {
      recording_failed_ = true;
      return false;
    }
    return CmdPresent(window);
  }

  bool CmdBuildBottomLevelAccelerationStructure(IBottomLevelAccelerationStructure *,
                                                const std::vector<rhi::AccelerationStructureGeometryBuffers> &,
                                                AccelerationStructureBuildMode) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *,
                                             const std::vector<rhi::RayTracingInstance> &,
                                             AccelerationStructureBuildMode) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdBindRayTracingProgram(IRayTracingProgram *) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdDispatchRays(uint32_t, uint32_t, uint32_t, IShaderBindingTable *) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdBindComputeProgram(IComputeProgram *) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdDispatch(uint32_t, uint32_t, uint32_t) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdBindGroup(uint32_t, IBindingGroup *) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdUploadImage(IImage *, const void *, uint64_t) override {
    recording_failed_ = true;
    return false;
  }

 private:
  bool PrepareRendering(const rhi::RenderingInfo &info) {
    auto next = info;
    if (next.color_attachments.size() > 8 || (next.color_attachments.empty() && !next.depth_stencil.image)) {
      recording_failed_ = true;
      return false;
    }
    std::vector<IImage *> images;
    auto validate = [&](IImage *image, bool depth, AttachmentLoadOp load, AttachmentStoreOp store) {
      auto native = rhi::QueryInterfacePtr<IBackendImage>(image).Get();
      if (!native || !rhi::QueryInterfacePtr<IImage>(native)->IsValid() ||
          (native->DeviceIdentity() != DeviceIdentity() || !native->SupportsQueue(OwnerQueue())) ||
          IsDepthFormat(image->GetDesc().format) != depth || load < ATTACHMENT_LOAD_OP_LOAD ||
          load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE || store > ATTACHMENT_STORE_OP_DISCARD)
        return false;
      for (auto previous : images)
        if (previous == image || previous->GetDesc().width != image->GetDesc().width ||
            previous->GetDesc().height != image->GetDesc().height)
          return false;
      images.push_back(image);
      return RetainResource(image);
    };
    for (auto color : next.color_attachments)
      if (!validate(color.color_attachment, false, color.load_op, color.store_op)) {
        recording_failed_ = true;
        return false;
      }
    const auto &ds = next.depth_stencil;
    if (ds.image &&
        (!validate(ds.image, true, ds.depth_load_op, ds.depth_store_op) ||
         ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD || ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD ||
         ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE || ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD ||
         !(ds.clear_value.depth >= 0 && ds.clear_value.depth <= 1))) {
      recording_failed_ = true;
      return false;
    }
    rendering_ = std::move(next);
    return true;
  }

  bool RenderingMatches(const IProgram *program) const {
    const auto &desc = program->GetDesc();
    if (desc.color_formats.size() != rendering_.color_attachments.size())
      return false;
    for (size_t i = 0; i < desc.color_formats.size(); ++i)
      if (desc.color_formats[i] != rendering_.color_attachments[i].color_attachment->GetDesc().format)
        return false;
    return desc.depth_stencil.format ==
           (rendering_.depth_stencil.image ? rendering_.depth_stencil.image->GetDesc().format : IMAGE_FORMAT_UNKNOWN);
  }

  void InitializeCommandContext(IBackendQueue *queue) {
    queue_ = queue;
  }

 public:
  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override {
    try {
      auto submission = queue_->Submit(ObjectPtr<IBackendCommandContext>::Retain(this), info);
      return rhi::QueryInterfacePtr<rhi::ISubmissionToken>(submission.Get()).Detach();
    } catch (const std::system_error &error) {
      ReportError(error.code().value(), error.what());
    }
    return nullptr;
  }

  // This mock rejects downloads; native download ownership is tested separately.
  void BindDownloads(rhi::ISubmissionToken *) override {
  }

  void ReleaseDownloads() override {
  }

  IDownloadToken *CmdDownloadBuffer(IBuffer *, uint64_t, uint64_t) override {
    recording_failed_ = true;
    return nullptr;
  }

  IDownloadToken *CmdDownloadImage(IImage *, const ImageDownloadRegion &) override {
    recording_failed_ = true;
    return nullptr;
  }

 private:
  template <typename T, std::enable_if_t<!std::is_convertible_v<T *, rhi::IObject *>, int> = 0>
  bool RetainResource(T *resource) {
    return RetainResource(rhi::QueryInterfacePtr<rhi::IObject>(resource).Get());
  }

  bool RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
    if (sealed_)
      return false;
#endif
    if (recording_failed_)
      return false;
    if (resources_.find(resource) != resources_.end())
      return true;
    resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));

    return true;
  }

 public:
  bool Seal() override {
    if (recording_failed_)
      return false;
#if !defined(NDEBUG)
    if (sealed_) {
      ReportError(-1, "Cannot submit a command context twice.");
      return false;
    }
#endif
    sealed_ = true;
    if (!EndRecording()) {
      recording_failed_ = true;
      return false;
    }
    return true;
  }

  IBackendQueue *OwnerQueue() const override {
    return queue_;
  }

  const void *DeviceIdentity() const override {
    return queue_->Owner();
  }

  void MarkSubmitted() override {
    submitted_ = true;
  }

  bool WasSubmitted() const override {
    return submitted_;
  }

  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdBindProgram(IProgram *program) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdBeginRendering(const rhi::RenderingInfo &info) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdEndRendering() override {
    recording_failed_ = true;
    ReportError(-1, "Drawing is not implemented for this backend.");
    return false;
  }

  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdSetViewport(const Viewport &viewport) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdSetScissor(const Scissor &scissor) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count,
               uint32_t first_vertex,
               uint32_t first_instance) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t vertex_offset,
                      uint32_t first_instance) override {
    recording_failed_ = true;
    ReportError(-1, "Drawing and buffers are not implemented for this backend.");
    return false;
  }

  bool CmdQueryCompactedSize(IAccelerationStructure *) override {
    recording_failed_ = true;
    return false;
  }

  bool CmdCompactAccelerationStructure(IAccelerationStructure *, IAccelerationStructure *) override {
    recording_failed_ = true;
    return false;
  }

 private:
  rhi::RenderingInfo rendering_{};
  bool recording_failed_{};
  bool sealed_{};
  IBackendQueue *queue_{};
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_{};
  bool submitted_{};
};

class TestQueue final : public backend::Object<rhi::IQueue, backend::IBackendQueue> {
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

 public:
  ObjectPtr<IBackendSubmissionToken> Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                            const SubmitInfo &info) override {
    if (!commands->Seal())
      return {};
    SubmitInfo native_info;
    native_info.waits.reserve(info.waits.size());
    for (const auto &wait : info.waits) {
      auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(wait.timeline).Get();
      if (!timeline || timeline->Owner() != device_) {
        ReportError(-1, "Submission waits require a timeline from this device.");
        return {};
      }
      // A submit waits for all dependencies; Vulkan requires unique semaphore handles.
      auto existing = std::find_if(native_info.waits.begin(), native_info.waits.end(),
                                   [&](const Event &event) { return event.timeline == wait.timeline; });
      if (existing == native_info.waits.end())
        native_info.waits.push_back(wait);
      else
        existing->value = std::max(existing->value, wait.value);
    }
    std::unique_lock<std::mutex> queue_lock(*mutex_);
    const Event signal{GetTimeline(), NextSerial()};
    auto completion = ObjectPtr<IBackendSubmissionToken>::Adopt(IBackendSubmissionToken::Create(signal));
    // Register before native submission, while the collector cannot take the list.
    auto entry = pending_.insert(pending_.end(), {commands, completion});
    IBackendCommandContext::SubmitResult result;
    try {
      commands->BindDownloads(rhi::QueryInterfacePtr<rhi::ISubmissionToken>(completion.Get()).Get());
      result = commands->SubmitNative(native_info, signal);
      if (result == IBackendCommandContext::SUBMIT_RESULT_REJECTED) {
        pending_.erase(entry);
        completion->Complete(SUBMISSION_STATUS_FAILED);
        commands->ReleaseDownloads();
        return {};
      }
      commands->MarkSubmitted();
    } catch (...) {
      if (!commands->WasSubmitted()) {
        pending_.erase(entry);
        completion->Complete(SUBMISSION_STATUS_FAILED);
        commands->ReleaseDownloads();
      }
      throw;
    }
    queue_lock.unlock();
    // The caller's context reference keeps it alive while the collector retires it.
    // Next-frame preparation belongs to the Window, not this submission's result.
    if (result == IBackendCommandContext::SUBMIT_RESULT_SUBMITTED)
      commands->UpdatePresentedImages();
    return result == IBackendCommandContext::SUBMIT_RESULT_SUBMITTED ? completion
                                                                     : ObjectPtr<IBackendSubmissionToken>{};
  }

  bool CollectCompleted(uint64_t completed, bool timeline_valid) override {
    std::list<Submission> work;
    {
      std::lock_guard<std::mutex> lock(*mutex_);
      work.splice(work.end(), pending_);
    }
    // Polling may submit a recovery signal; releasing resources may acquire other
    // backend locks. Neither operation runs under the submission mutex.
    for (auto it = work.begin(); it != work.end();) {
      const auto status = it->commands->PollCompletion(
          rhi::QueryInterfacePtr<ISubmissionToken>(it->completion.Get())->GetEvent(), completed, timeline_valid);
      if (status == SUBMISSION_STATUS_PENDING) {
        ++it;
        continue;
      }
      it->commands->ReleaseDownloads();
      it->commands.Reset();
      it->completion->Complete(status);
      it = work.erase(it);
    }
    std::lock_guard<std::mutex> lock(*mutex_);
    pending_.splice(pending_.begin(), work);
    return pending_.empty();
  }

 private:
  void InitializeQueue(IBackendDevice *device,
                       QueueType type,
                       std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>()) {
    device_ = device;
    type_ = type;
    mutex_ = std::move(mutex);
  }

 public:
  std::mutex &QueueMutex() override {
    return *mutex_;
  }

  uint64_t NextSerial() override {
    return ++serial_;
  }

  IBackendDevice *Owner() const override {
    return device_;
  }

  QueueType GetType() const override {
    return type_;
  }

 private:
  struct Submission {
    ObjectPtr<IBackendCommandContext> commands;
    ObjectPtr<IBackendSubmissionToken> completion;
  };

  std::list<Submission> pending_{};
  IBackendDevice *device_{};
  QueueType type_{};
  // Logical queues that share a native queue also share its host synchronization.
  std::shared_ptr<std::mutex> mutex_{};
  uint64_t serial_{};
};

class TestDevice final : public backend::Object<rhi::IDevice, backend::IBackendDevice> {
 public:
  explicit TestDevice(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
    InitializeDevice({}, {});

    for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST})
      queues_[type] = ObjectPtr<backend::IBackendQueue>::Adopt(new TestQueue(this, type, gate_));
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

 public:
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &) override {
    ReportError(-1, "Acceleration structures are unavailable for this backend.");
    return nullptr;
  }

  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &) override {
    ReportError(-1, "Acceleration structures are unavailable for this backend.");
    return nullptr;
  }

  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &) override {
    ReportError(-1, "Ray tracing pipelines are unavailable for this backend.");
    return nullptr;
  }

 private:
  void InitializeDevice(PhysicalDevice info, ICore *core, BackendAPI api = BACKEND_API_DEFAULT) {
    api_ = api;
    core_ = core;
    info_ = std::move(info);
  }

 public:
  rhi::IImage *CreateImage(const ImageDesc &) override {
    ReportError(-1, "Independent images are not implemented for this backend.");
    return nullptr;
  }

  rhi::IBuffer *CreateBuffer(const BufferDesc &) override {
    ReportError(-1, "Buffers are not implemented for this backend.");
    return nullptr;
  }

  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &) override {
    ReportError(-1, "Compute programs are not implemented for this backend.");
    return nullptr;
  }

  rhi::IProgram *CreateProgram(const ProgramDesc &) override {
    ReportError(-1, "Graphics programs are not implemented for this backend.");
    return nullptr;
  }

  rhi::IShader *CreateShader(const ShaderDesc &desc) override {
    if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
        desc.stage > SHADER_STAGE_CALLABLE)
      return nullptr;
    return backend::CreateShader(desc, api_);
  }

  ISampler *CreateSampler(const SamplerDesc &) override {
    ReportError(-1, "Samplers are not implemented for this backend.");
    return nullptr;
  }

  IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &) override {
    ReportError(-1, "Binding groups are not implemented for this backend.");
    return nullptr;
  }

 private:
  void DestroyDevice() {
    ShutdownQueues();
  }

 public:
  std::string Name() const override {
    return info_.name;
  }

  bool SupportRayTracingPipeline() const override {
    return info_.ray_tracing_pipeline_feature;
  }

  bool SupportRayQuery() const override {
    return info_.ray_query_feature;
  }

  bool SupportGeometryShader() const override {
    return info_.geometry_shader_feature;
  }

  rhi::IWindow *CreateGLFWWindow(const WindowDesc &) override {
    ReportError(-1, "Window creation is unavailable.");
    return nullptr;
  }

  rhi::IQueue *GetQueue(QueueType type) override {
    if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
      return nullptr;
    return rhi::QueryInterfacePtr<rhi::IQueue>(queues_[type].Get()).Detach();
  }

  const PhysicalDevice &GetPhysicalDevice() const noexcept override {
    return info_;
  }

 private:
  ICore *OwnerCore() const {
    return core_;
  }

  void StartCollector() {
    collector_ = std::thread([this] { Collect(); });
  }

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all) const override {
    if (events.empty())
      return wait_all;
    std::vector<Event> unique;
    unique.reserve(events.size());
    for (const auto &event : events) {
      auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(event.timeline).Get();
      if (!timeline || timeline->Owner() != this) {
        ReportError(-1, "Host waits require timelines from this device.");
        return false;
      }
      auto found = std::find_if(unique.begin(), unique.end(),
                                [&](const Event &other) { return other.timeline == event.timeline; });
      if (found == unique.end())
        unique.push_back(event);
      else
        found->value = wait_all ? std::max(found->value, event.value) : std::min(found->value, event.value);
    }
    return WaitNativeEvents(unique, wait_all);
  }

 private:
  void Collect() {
    std::vector<Event> next;
    next.reserve(queues_.size());
    for (;;) {
      next.clear();
      bool empty = true;
      for (const auto &queue : queues_) {
        if (!queue)
          continue;
        uint64_t completed = 0;
        const bool timeline_valid =
            rhi::QueryInterfacePtr<rhi::ITimeline>(queue->GetTimeline())->GetCompletedValue(completed);
        if (timeline_valid && completed != UINT64_MAX)
          next.push_back({rhi::QueryInterfacePtr<rhi::ITimeline>(queue->GetTimeline()).Get(), completed + 1});
        if (!queue->CollectCompleted(completed, timeline_valid))
          empty = false;
      }
      if (stopping_ && empty)
        return;
      WaitForProgress(next);
    }
  }

  void ShutdownQueues() {
    if (!collector_.joinable())
      return;
    stopping_ = true;
    InterruptCollector();
    collector_.join();
  }

  std::array<ObjectPtr<IBackendQueue>, 2> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  ICore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
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
  ASSERT_TRUE(native->CmdUseResource(rhi::QueryInterfacePtr<rhi::IObject>(image.Get()).Get()));
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
  ASSERT_TRUE(context->CmdUseResource(rhi::QueryInterfacePtr<rhi::IObject>(image.Get()).Get()));
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
