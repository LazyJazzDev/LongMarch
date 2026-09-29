#include "grassland/graphics/window.h"

#include "grassland/graphics/buffer.h"
#include "grassland/graphics/command_context.h"
#include "grassland/graphics/core.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/program.h"
#include "grassland/graphics/shader.h"
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

#if defined(__APPLE__) && !defined(LONGMARCH_HEADLESS)
#include "grassland/graphics/window_gestures.h"
#endif

namespace grassland::graphics {

#ifndef LONGMARCH_HEADLESS
ImGuiLinearColors::ImGuiLinearColors(bool enabled) {
  if (!enabled || !ImGui::GetDrawData())
    return;
  for (auto *list : ImGui::GetDrawData()->CmdLists) {
    for (auto &vertex : list->VtxBuffer) {
      colors_.emplace_back(&vertex, vertex.col);
      auto color = ImGui::ColorConvertU32ToFloat4(vertex.col);
      auto linear = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
      vertex.col = ImGui::ColorConvertFloat4ToU32({linear(color.x), linear(color.y), linear(color.z), color.w});
    }
  }
}

ImGuiLinearColors::~ImGuiLinearColors() {
  for (const auto &entry : colors_)
    entry.first->col = entry.second;
}
#endif

struct Window::HDRPresentation {
  Core *core{};
  std::unique_ptr<Image> composition, aligned;
  std::unique_ptr<Shader> shader;
  std::unique_ptr<ComputeProgram> program;
  std::unique_ptr<Buffer> settings;
};

DisplayBrightness Window::QueryDisplayBrightness() const {
  DisplayBrightness result;
#ifdef _WIN32
  if (!window_)
    return result;
  MONITORINFOEXW monitor{};
  monitor.cbSize = sizeof(monitor);
  const auto native_monitor = MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST);
  if (!GetMonitorInfoW(native_monitor, &monitor))
    return result;
  UINT32 path_count{}, mode_count{};
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS)
    return result;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) !=
      ERROR_SUCCESS)
    return result;
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
      return result;
    result.hdr_enabled = true;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(white), path.targetInfo.adapterId,
                    path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS && white.SDRWhiteLevel > 0) {
      result.hdr_reference_white_scale = white.SDRWhiteLevel / 1000.0f;
      result.sdr_white_nits = 80.0f * result.hdr_reference_white_scale;
      result.reference_white_known = true;
    }
    // Match the window's display across all adapters, including hybrid GPUs.
    // This is a Windows display query shared by Vulkan and D3D12.
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
      return result;
    for (UINT adapter_index = 0;; ++adapter_index) {
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
      if (FAILED(factory->EnumAdapters1(adapter_index, &adapter)))
        break;
      for (UINT output_index = 0;; ++output_index) {
        Microsoft::WRL::ComPtr<IDXGIOutput> output;
        if (FAILED(adapter->EnumOutputs(output_index, &output)))
          break;
        Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
        DXGI_OUTPUT_DESC1 desc{};
        if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)) || desc.Monitor != native_monitor)
          continue;
        if (!std::isfinite(desc.MaxLuminance) || desc.MaxLuminance <= 0.0f)
          return result;
        if (result.reference_white_known && result.sdr_white_nits > 0.0f) {
          const float ratio = desc.MaxLuminance / result.sdr_white_nits;
          if (std::isfinite(ratio)) {
            result.hdr_headroom = std::max(1.0f, ratio);
          }
        }
        return result;
      }
    }
    return result;
  }
#endif
  return result;
}

DisplayBrightness Window::GetDisplayBrightness() {
  if (std::chrono::steady_clock::now() - brightness_query_time_ >= std::chrono::milliseconds(500))
    RefreshDisplayBrightness();
  return display_brightness_;
}

void Window::RefreshDisplayBrightness() {
  auto next = QueryDisplayBrightness();
  brightness_query_time_ = std::chrono::steady_clock::now();
  const auto previous = display_brightness_;
  display_brightness_ = next;
  if (previous.sdr_white_nits != next.sdr_white_nits ||
      previous.hdr_reference_white_scale != next.hdr_reference_white_scale ||
      previous.hdr_headroom != next.hdr_headroom || previous.reference_white_known != next.reference_white_known ||
      previous.hdr_enabled != next.hdr_enabled)
    display_brightness_event_.InvokeCallbacks(display_brightness_);
}

void Window::SetHDRBrightnessAlignment(bool enabled) {
  align_hdr_brightness_ = enabled;
}

float Window::HDRReferenceWhiteScale() {
  const auto brightness = GetDisplayBrightness();
  return enable_hdr_ && align_hdr_brightness_ ? brightness.hdr_reference_white_scale : 1.0f;
}

Image *Window::PrepareHDRComposition(Core *core, Extent2D extent) {
  if (!enable_hdr_)
    return nullptr;
  if (!hdr_presentation_) {
    auto pending = std::make_unique<HDRPresentation>();
    auto &p = *pending;
    p.core = core;
    const char *source = R"(
Texture2D<float4> source_image : register(t0, space0);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> output_image : register(u0, space1);
cbuffer Settings : register(b0, space2) { float white_scale; float pq_output; float white_nits; float padding; };
float3 EncodePQ(float3 nits) {
  float3 p = pow(saturate(nits / 10000.0), 2610.0 / 16384.0);
  return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) /
             (1.0 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
}
[numthreads(8,8,1)] void Main(uint3 id : SV_DispatchThreadID) {
  uint width, height;
  output_image.GetDimensions(width, height);
  if (id.x >= width || id.y >= height) return;
  float4 color = source_image.Load(int3(id.xy, 0));
  float3 rgb = max(color.rgb, 0.0);
  if (pq_output != 0.0) {
    // Linear BT.709/D65 to BT.2020/D65, then absolute-luminance ST 2084.
    rgb = mul(float3x3(0.627404, 0.329283, 0.043313,
                       0.069097, 0.919540, 0.011362,
                       0.016391, 0.088013, 0.895595), rgb);
    rgb = EncodePQ(rgb * white_nits);
  } else {
    rgb = min(rgb * white_scale, 65504.0);
  }
  output_image[id.xy] = float4(rgb, color.a);
}
)";
    if (core->CreateShader(source, "Main", "cs_6_0", &p.shader) ||
        core->CreateComputeProgram(p.shader.get(), &p.program))
      throw std::runtime_error("Cannot create HDR reference-white presentation shader");
    p.program->AddResourceBinding(RESOURCE_TYPE_IMAGE, 1);
    p.program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_IMAGE, 1);
    p.program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    p.program->Finalize();
    if (core->CreateBuffer(16, BUFFER_TYPE_STATIC, &p.settings))
      throw std::runtime_error("Cannot create HDR presentation settings");
    hdr_presentation_ = std::move(pending);
  }
  auto &p = *hdr_presentation_;
  if (p.core != core)
    throw std::invalid_argument("HDR presentation belongs to a different graphics core");
  if (!p.composition || p.composition->Extent().width != extent.width ||
      p.composition->Extent().height != extent.height) {
    core->WaitGPU();
    core->CreateImage(extent.width, extent.height, IMAGE_FORMAT_R16G16B16A16_SFLOAT, &p.composition);
    core->CreateImage(extent.width, extent.height, IMAGE_FORMAT_R16G16B16A16_SFLOAT, &p.aligned);
  }
  return p.composition.get();
}

Image *Window::AlignHDRComposition(CommandContext *commands) {
  if (!hdr_presentation_ || !hdr_presentation_->composition)
    throw std::logic_error("Prepare HDR composition before aligning it");
  auto &p = *hdr_presentation_;
  // PQ content reference white matches the WSI color-description default.
  // Output reference white is mapped by the compositor, not applied here.
  const float settings[4] = {HDRReferenceWhiteScale(), UsesPQOutput() ? 1.0f : 0.0f, 203.0f, 0};
  p.settings->UploadData(settings, sizeof(settings));
  commands->CmdBindComputeProgram(p.program.get());
  commands->CmdBindResources(0, {p.composition.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {p.aligned.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, {p.settings.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch((p.aligned->Extent().width + 7) / 8, (p.aligned->Extent().height + 7) / 8, 1);
  return p.aligned.get();
}

#ifndef LONGMARCH_HEADLESS
Window::Window(int width, int height, const std::string &title, bool fullscreen, bool resizable, bool enable_hdr)
    : enable_hdr_(enable_hdr) {
  if (!Core::InitializeGLFW())
    throw std::runtime_error("Failed to initialize GLFW");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  if (fullscreen) {
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  } else {
    if (!resizable) {
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    } else {
      glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    }
  }

  window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

  if (!window_) {
    throw std::runtime_error("Failed to create GLFW window");
  }

#ifdef __APPLE__
  magnify_monitor_ = detail::InstallMagnifyEvents(this);
#endif
  resize_size_ = GetSize();
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow *window, int width, int height) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->framebuffer_resize_event_.InvokeCallbacks(width, height);
  });
  glfwSetCursorEnterCallback(window_, [](GLFWwindow *window, int entered) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->cursor_enter_event_.InvokeCallbacks(entered == GLFW_TRUE);
  });
  glfwSetWindowFocusCallback(window_, [](GLFWwindow *window, int focused) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->focus_event_.InvokeCallbacks(focused == GLFW_TRUE);
  });
  glfwSetWindowSizeCallback(window_, [](GLFWwindow *window, int width, int height) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->NotifyResize();
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow *window, int button, int action, int mods) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    p_window->mouse_button_event_.InvokeCallbacks(button, action, mods, x, y);
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow *window, double x, double y) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->mouse_move_event_.InvokeCallbacks(x, y);
  });
  glfwSetScrollCallback(window_, [](GLFWwindow *window, double xoffset, double yoffset) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->scroll_event_.InvokeCallbacks(xoffset, yoffset);
  });
  glfwSetKeyCallback(window_, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->key_event_.InvokeCallbacks(key, scancode, action, mods);
  });
  glfwSetCharCallback(window_, [](GLFWwindow *window, unsigned int codepoint) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->char_event_.InvokeCallbacks(codepoint);
  });
  glfwSetDropCallback(window_, [](GLFWwindow *window, int count, const char **paths) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->drop_event_.InvokeCallbacks(count, paths);
  });
}
#endif

Window::Window(Hosted, int width, int height, const std::string &title)
    : hosted_(true),
      hosted_size_(width, height),
      hosted_framebuffer_(width, height),
      hosted_title_(title),
      enable_hdr_(false) {
}

void Window::UpdateHostedSize(glm::ivec2 logical, glm::ivec2 framebuffer) {
  if (!hosted_)
    throw std::logic_error("Expected hosted window");
  if (logical != hosted_size_) {
    hosted_size_ = logical;
    resize_event_.InvokeCallbacks(logical.x, logical.y);
  }
  if (framebuffer != hosted_framebuffer_) {
    hosted_framebuffer_ = framebuffer;
    framebuffer_resize_event_.InvokeCallbacks(framebuffer.x, framebuffer.y);
  }
}

void Window::SendPointer(double x, double y) {
  hosted_cursor_ = {x, y};
  mouse_move_event_.InvokeCallbacks(x, y);
}

void Window::SendMouseButton(int button, int action, int mods) {
  if (action)
    hosted_buttons_.insert(button);
  else
    hosted_buttons_.erase(button);
  mouse_button_event_.InvokeCallbacks(button, action, mods, hosted_cursor_.x, hosted_cursor_.y);
}

void Window::SendKey(int key, int action, int mods) {
  if (action)
    hosted_keys_.insert(key);
  else
    hosted_keys_.erase(key);
  key_event_.InvokeCallbacks(key, 0, action, mods);
}

void Window::SendFocus(bool focused) {
  hosted_focused_ = focused;
  if (!focused) {
    hosted_keys_.clear();
    hosted_buttons_.clear();
  }
  focus_event_.InvokeCallbacks(focused);
}

Window::~Window() {
  CloseWindow();
}

int Window::GetWidth() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    return width;
  }
#endif
  return hosted_size_.x;
}

int Window::GetHeight() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    return height;
  }
#endif
  return hosted_size_.y;
}

glm::ivec2 Window::GetSize() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glm::ivec2 size;
    glfwGetWindowSize(window_, &size.x, &size.y);
    return size;
  }
#endif
  return hosted_size_;
}

glm::ivec2 Window::GetFramebufferSize() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glm::ivec2 size;
    glfwGetFramebufferSize(window_, &size.x, &size.y);
    return size;
  }
#endif
  return hosted_framebuffer_;
}

glm::dvec2 Window::GetCursorPosition() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glm::dvec2 position;
    glfwGetCursorPos(window_, &position.x, &position.y);
    return position;
  }
#endif
  return hosted_cursor_;
}

bool Window::IsKeyDown(int key) const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    return glfwGetKey(window_, key) == GLFW_PRESS;
  }
#endif
  return hosted_keys_.count(key);
}

bool Window::IsMouseButtonDown(int button) const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    return glfwGetMouseButton(window_, button) == GLFW_PRESS;
  }
#endif
  return hosted_buttons_.count(button);
}

bool Window::IsFocused() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    return glfwGetWindowAttrib(window_, GLFW_FOCUSED) == GLFW_TRUE;
  }
#endif
  return hosted_focused_;
}

void Window::Focus() {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glfwFocusWindow(window_);
  }
#endif
  SendFocus(true);
}

void Window::RequestClose() {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }
#endif
  hosted_closed_ = true;
}

glm::ivec2 Window::GetPosition() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    // Wayland intentionally does not expose global window coordinates.
#if defined(__linux__) && defined(GLFW_PLATFORM)
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
      return {0, 0};
#endif
    glm::ivec2 position{};
    glfwGetWindowPos(window_, &position.x, &position.y);
    return position;
  }
#endif
  return {0, 0};
}

void Window::SetPosition(int x, int y) {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
#if defined(__linux__) && defined(GLFW_PLATFORM)
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
      return;
#endif
    glfwSetWindowPos(window_, x, y);
    return;
  }
#endif
}

glm::ivec4 Window::GetFrameSize() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glm::ivec4 frame{};
    glfwGetWindowFrameSize(window_, &frame.x, &frame.y, &frame.z, &frame.w);
    return frame;
  }
#endif
  return {0, 0, 0, 0};
}

glm::ivec4 Window::GetMonitorWorkArea() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
#if defined(__linux__) && defined(GLFW_PLATFORM)
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
      // GLFW cannot locate an ordinary Wayland window in global coordinates.
      // Use the primary output only as a sizing hint; let the compositor place it.
      glm::ivec4 area{0, 0, GetWidth(), GetHeight()};
      if (auto *monitor = glfwGetPrimaryMonitor())
        glfwGetMonitorWorkarea(monitor, &area.x, &area.y, &area.z, &area.w);
      return area;
    }
#endif
    const auto position = GetPosition();
    const auto size = GetSize();
    int count = 0;
    auto **monitors = glfwGetMonitors(&count);
    auto *monitor = glfwGetPrimaryMonitor();
    int64_t best_overlap = 0;
    for (int i = 0; i < count; ++i) {
      glm::ivec4 area;
      glfwGetMonitorWorkarea(monitors[i], &area.x, &area.y, &area.z, &area.w);
      const auto overlap =
          glm::max(glm::ivec2{0}, glm::min(position + size, glm::ivec2{area.x + area.z, area.y + area.w}) -
                                      glm::max(position, glm::ivec2{area.x, area.y}));
      const int64_t pixels = int64_t(overlap.x) * overlap.y;
      if (pixels > best_overlap) {
        best_overlap = pixels;
        monitor = monitors[i];
      }
    }
    if (!monitor)
      return {position.x, position.y, size.x, size.y};
    glm::ivec4 area;
    glfwGetMonitorWorkarea(monitor, &area.x, &area.y, &area.z, &area.w);
    return area;
  }
#endif
  return {0, 0, hosted_size_.x, hosted_size_.y};
}

void Window::PollEvents() {
#ifndef LONGMARCH_HEADLESS
  glfwPollEvents();
#endif
}

void Window::SetTitle(const std::string &title) {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glfwSetWindowTitle(window_, title.c_str());
    return;
  }
#endif
  hosted_title_ = title;
}

std::string Window::GetTitle() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    return glfwGetWindowTitle(window_);
  }
#endif
  return hosted_title_;
}

void Window::NotifyResize() {
  if (!window_)
    return;
  const auto size = GetSize();
  if (size == resize_size_)
    return;
  // Wayland programmatic resizing may omit the logical-size callback.
  // Deduplicate both native callbacks and Resize() by logical size; pixel-only
  // scaling remains exclusively a FramebufferResizeEvent.
  resize_size_ = size;
  resize_event_.InvokeCallbacks(size.x, size.y);
}

void Window::Resize(int new_width, int new_height) {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    glfwSetWindowSize(window_, new_width, new_height);
    // GLFW's Wayland backend can omit the logical-size callback here.
    // NotifyResize deduplicates against any native callback already delivered.
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
      NotifyResize();
    return;
  }
#endif
  UpdateHostedSize({new_width, new_height}, {new_width, new_height});
}

void Window::CloseWindow() {
  hdr_presentation_.reset();
  if (hosted_) {
    hosted_closed_ = true;
    return;
  }
#ifndef LONGMARCH_HEADLESS
#ifdef __APPLE__
  detail::RemoveMagnifyEvents(magnify_monitor_);
  magnify_monitor_ = nullptr;
#endif
  glfwDestroyWindow(window_);
  window_ = nullptr;
#endif
}

bool Window::ShouldClose() const {
#ifndef LONGMARCH_HEADLESS
  if (!hosted_) {
    return glfwWindowShouldClose(window_);
  }
#endif
  return hosted_closed_;
}

int Window::SetHDR(bool enable_hdr) {
  if (!window_)
    return -1;
  const bool previous = enable_hdr_;
  try {
    enable_hdr_ = enable_hdr;
    RefreshDisplayBrightness();
    resize_event_.InvokeCallbacks(GetWidth(), GetHeight());
    return 0;
  } catch (const std::exception &error) {
    enable_hdr_ = previous;
    LogError("Failed to change HDR presentation: {}", error.what());
    return -1;
  }
}

}  // namespace grassland::graphics
