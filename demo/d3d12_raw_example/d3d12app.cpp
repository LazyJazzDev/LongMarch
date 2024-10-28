#include "d3d12app.h"

// Include GLFW native window handle
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

namespace d3d12 {

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

std::string CheckRayTracingSupport(IDXGIAdapter1 *adapter) {
  ComPtr<ID3D12Device5> device;
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 featureSupportData = {};
  ThrowIfFailed(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0,
                                  IID_PPV_ARGS(&device)));
  ThrowIfFailed(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
                                            &featureSupportData,
                                            sizeof(featureSupportData)));
  switch (featureSupportData.RaytracingTier) {
    case D3D12_RAYTRACING_TIER_NOT_SUPPORTED:
      return "Raytracing not supported";
    case D3D12_RAYTRACING_TIER_1_0:
      return "Raytracing Tier 1.0 supported";
    case D3D12_RAYTRACING_TIER_1_1:
      return "Raytracing Tier 1.1 supported";
    default:
      return "Unknown Raytracing Tier";
  }
}

void Application::CreateWindowAssets() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  glfw_window_ = glfwCreateWindow(2560, 1440, "D3D12", nullptr, nullptr);

  // Get HWND from GLFW window
  HWND hwnd = glfwGetWin32Window(glfw_window_);
  HINSTANCE hinstance = GetModuleHandle(nullptr);

  UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the
  // active device.
  {
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }
#endif

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<IDXGIFactory6> factory6;
  if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
    for (uint32_t adapter_index = 0;
         SUCCEEDED(factory6->EnumAdapterByGpuPreference(
             adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
             IID_PPV_ARGS(&adapter)));
         adapter_index++) {
      DXGI_ADAPTER_DESC1 desc;
      adapter->GetDesc1(&desc);
      LogInfo("Device Name: {}", WStringToString(desc.Description));
      LogInfo("- Raytracing Support: {}",
              CheckRayTracingSupport(adapter.Get()));
    }
  }
}

void Application::DestroyWindowAssets() {
  glfwDestroyWindow(glfw_window_);

  glfwTerminate();
}

}  // namespace d3d12
