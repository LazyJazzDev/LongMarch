#include "d3d12app.h"

// Include GLFW native window handle
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

namespace D3D12 {

Application::Application() {
}

void Application::Run() {
  OnInit();
  while (!glfwWindowShouldClose(glfw_window_)) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }
  OnClose();
}

void Application::OnInit() {
  CreateWindowAssets();
}

void Application::OnUpdate() {
}

void Application::OnRender() {
}

void Application::OnClose() {
  DestroyWindowAssets();
}

void Application::CreateWindowAssets() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  glfw_window_ = glfwCreateWindow(2560, 1440, "D3D12", nullptr, nullptr);

  // Get HWND from GLFW window
  d3d12::CreateDXGIFactory(&factory_);
}

void Application::DestroyWindowAssets() {
  glfwDestroyWindow(glfw_window_);

  glfwTerminate();
}

}  // namespace D3D12
