#include "d3d12app.h"

// Include GLFW native window handle
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "grassland/d3d12/device.h"
#include "grassland/util/vendor_id.h"

namespace D3D12 {

namespace {
#include "built_in_shaders.inl"
}

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
  uint32_t back_buffer_index =
      swap_chain_->Handle()->GetCurrentBackBufferIndex();

  command_allocator_->ResetCommandRecord(command_list_.get());

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      swap_chain_->BackBuffer(back_buffer_index), D3D12_RESOURCE_STATE_PRESENT,
      D3D12_RESOURCE_STATE_RENDER_TARGET);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  auto rtv_handle = swap_chain_->RTVCPUHandle(back_buffer_index);

  const float clear_color[] = {0.6f, 0.7f, 0.8f, 1.0f};

  command_list_->Handle()->ClearRenderTargetView(rtv_handle, clear_color, 0,
                                                 nullptr);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      swap_chain_->BackBuffer(back_buffer_index),
      D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  command_list_->Handle()->Close();

  ID3D12CommandList *command_lists[] = {command_list_->Handle()};
  command_queue_->Handle()->ExecuteCommandLists(1, command_lists);

  swap_chain_->Handle()->Present(1, 0);

  fence_->Signal(command_queue_.get());
  fence_->Wait();
}

void Application::OnClose() {
  DestroyWindowAssets();
}

void Application::CreateWindowAssets() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  glfw_window_ = glfwCreateWindow(2560, 1440, "D3D12", nullptr, nullptr);

  CreateDXGIFactory(&factory_);

  factory_->CreateDevice(DeviceFeatureRequirement{}, -1, &device_);
  LogInfo("Device: {}", device_->Adapter().Name());
  LogInfo("- Vendor: {}", PCIVendorIDToName(device_->Adapter().VendorID()));
  LogInfo("- Device Feature Level: {}.{}",
          uint32_t(device_->FeatureLevel()) >> 12,
          (uint32_t(device_->FeatureLevel()) >> 8) & 0xf);

  device_->CreateCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_queue_);

  device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  &command_allocator_);

  command_allocator_->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        &command_list_);

  factory_->CreateSwapChain(*command_queue_, glfwGetWin32Window(glfw_window_),
                            frame_count, &swap_chain_);

  device_->CreateFence(&fence_);
}

void Application::DestroyWindowAssets() {
  glfwDestroyWindow(glfw_window_);

  glfwTerminate();
}

}  // namespace D3D12
