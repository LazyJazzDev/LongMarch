#include "d3d12app.h"

// Include GLFW native window handle
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "grassland/d3d12/device.h"
#include "grassland/util/vendor_id.h"

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
  CreateDXGIFactory(&factory_);

  factory_->CreateDevice(DeviceFeatureRequirement{}, -1, &device_);
  LogInfo("Device: {}", device_->Adapter().Name());
  LogInfo("- Vendor: {}", PCIVendorIDToName(device_->Adapter().VendorID()));
  LogInfo("- Device Feature Level: {}.{}",
          uint32_t(device_->FeatureLevel()) >> 12,
          (uint32_t(device_->FeatureLevel()) >> 8) & 0xf);
}

void Application::DestroyWindowAssets() {
  glfwDestroyWindow(glfw_window_);

  glfwTerminate();
}

}  // namespace D3D12
