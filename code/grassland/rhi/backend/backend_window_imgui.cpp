#include "grassland/rhi/backend/backend_window_imgui.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>

#include "imgui_impl_glfw.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace grassland::rhi::backend {
namespace {
#ifdef _WIN32
constexpr wchar_t context_property[] = L"LongMarch.RHI.ImGuiContext";
constexpr wchar_t procedure_property[] = L"LongMarch.RHI.ImGuiWndProc";

LRESULT CALLBACK ImGuiWindowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  auto context = reinterpret_cast<ImGuiContext *>(GetPropW(window, context_property));
  auto procedure = reinterpret_cast<WNDPROC>(GetPropW(window, procedure_property));
  ImGuiContextScope scope(context);
  return CallWindowProcW(procedure, window, message, wparam, lparam);
}
#endif

}  // namespace

ImGuiContextScope::ImGuiContextScope(ImGuiContext *context) : previous_(ImGui::GetCurrentContext()) {
  ImGui::SetCurrentContext(context);
}

ImGuiContextScope::~ImGuiContextScope() {
  ImGui::SetCurrentContext(previous_);
}

WindowImGui::WindowImGui(IDevice *device) : device_(device) {
}

std::unique_ptr<WindowImGui> WindowImGui::Create(IDevice *device,
                                                 GLFWwindow *window,
                                                 const char *font_file_path,
                                                 float font_size) {
  auto result = std::unique_ptr<WindowImGui>(new WindowImGui(device));
  if (!result->Initialize(window, font_file_path, font_size))
    return {};
  return result;
}

bool WindowImGui::Initialize(GLFWwindow *window, const char *font_file_path, float font_size) {
  renderer_ = ImGuiRenderer::Create(device_, font_file_path, font_size);
  if (!renderer_)
    return false;
  ImGuiContextScope scope(renderer_->Context());
  platform_initialized_ = ImGui_ImplGlfw_InitForOther(window, false);
  if (!platform_initialized_)
    return false;
#ifdef _WIN32
  // The GLFW backend installs a Win32 hook even with install_callbacks=false.
  // Select this HWND's context before that hook reads BackendPlatformUserData;
  // native messages also arrive outside PollEvents and ImGui frame recording.
  auto native = static_cast<HWND>(ImGui::GetMainViewport()->PlatformHandleRaw);
  auto procedure = reinterpret_cast<WNDPROC>(GetWindowLongPtrW(native, GWLP_WNDPROC));
  if (!SetPropW(native, context_property, renderer_->Context()))
    return false;
  if (!SetPropW(native, procedure_property, reinterpret_cast<HANDLE>(procedure))) {
    RemovePropW(native, context_property);
    return false;
  }
  SetLastError(0);
  if (!SetWindowLongPtrW(native, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(ImGuiWindowProc)) && GetLastError()) {
    RemovePropW(native, procedure_property);
    RemovePropW(native, context_property);
    return false;
  }
#endif
  return true;
}

WindowImGui::~WindowImGui() {
  if (!renderer_)
    return;
  auto previous = ImGui::GetCurrentContext();
  ImGui::SetCurrentContext(renderer_->Context());
  if (platform_initialized_) {
#ifdef _WIN32
    auto native = static_cast<HWND>(ImGui::GetMainViewport()->PlatformHandleRaw);
    if (auto procedure = RemovePropW(native, procedure_property))
      SetWindowLongPtrW(native, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(procedure));
    RemovePropW(native, context_property);
#endif
    ImGui_ImplGlfw_Shutdown();
  }
  ImGui::SetCurrentContext(previous == renderer_->Context() ? nullptr : previous);
}

ImGuiContext *WindowImGui::Context() const {
  return renderer_->Context();
}

bool WindowImGui::Update(const std::function<void()> &callback) {
  ImGuiContextScope scope(renderer_->Context());
  ImGui_ImplGlfw_NewFrame();
  return renderer_->Update(callback);
}

bool WindowImGui::Record(ICommandContext *commands,
                         IImage *image,
                         ColorSpace color_space,
                         float reference_white_scale) {
  return renderer_->Record(commands, image, color_space, reference_white_scale);
}
}  // namespace grassland::rhi::backend
