#include "d3d12app.h"

#include "glm/gtc/matrix_transform.hpp"

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
  CreatePipelineAssets();
}

void Application::OnUpdate() {
}

void Application::OnRender() {
  uint32_t back_buffer_index = swap_chain_->Handle()->GetCurrentBackBufferIndex();

  command_allocator_->ResetCommandRecord(command_list_.get());

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      swap_chain_->BackBuffer(back_buffer_index), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  auto rtv_handle = swap_chain_->RTVCPUHandle(back_buffer_index);

  const float clear_color[] = {0.6f, 0.7f, 0.8f, 1.0f};

  command_list_->Handle()->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(swap_chain_->BackBuffer(back_buffer_index),
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
  DestroyPipelineAssets();
  DestroyWindowAssets();
}

void Application::CreateWindowAssets() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  glfw_window_ = glfwCreateWindow(800, 600, "D3D12 Hello Ray Tracing", nullptr, nullptr);

  CreateDXGIFactory({}, &factory_);

  factory_->CreateDevice(DeviceFeatureRequirement{true}, -1, &device_);
  LogInfo("Device: {}", device_->Adapter().Name());
  LogInfo("- Vendor: {}", PCIVendorIDToName(device_->Adapter().VendorID()));
  LogInfo("- Device Feature Level: {}.{}", uint32_t(device_->FeatureLevel()) >> 12,
          (uint32_t(device_->FeatureLevel()) >> 8) & 0xf);

  device_->CreateCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_queue_);

  device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocator_);

  command_allocator_->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_list_);

  factory_->CreateSwapChain(*command_queue_, glfwGetWin32Window(glfw_window_), frame_count, &swap_chain_);

  device_->CreateFence(&fence_);
}

void Application::DestroyWindowAssets() {
  glfwDestroyWindow(glfw_window_);

  glfwTerminate();
}

void Application::CreatePipelineAssets() {
  std::vector<glm::vec3> vertices = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};

  std::unique_ptr<Buffer> staging_vertex_buffer;
  std::unique_ptr<Buffer> staging_index_buffer;

  device_->CreateBuffer(vertices.size() * sizeof(glm::vec3), D3D12_HEAP_TYPE_UPLOAD, &staging_vertex_buffer);
  device_->CreateBuffer(indices.size() * sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD, &staging_index_buffer);

  device_->CreateBuffer(vertices.size() * sizeof(glm::vec3), D3D12_HEAP_TYPE_DEFAULT, &vertex_buffer_);
  device_->CreateBuffer(indices.size() * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, &index_buffer_);
  std::memcpy(staging_vertex_buffer->Map(), vertices.data(), vertices.size() * sizeof(glm::vec3));
  staging_vertex_buffer->Unmap();
  std::memcpy(staging_index_buffer->Map(), indices.data(), indices.size() * sizeof(uint32_t));
  staging_index_buffer->Unmap();

  command_queue_->SingleTimeCommand(
      [&staging_vertex_buffer, &staging_index_buffer, this](ID3D12GraphicsCommandList *cmd_list) {
        CopyBuffer(cmd_list, staging_vertex_buffer.get(), vertex_buffer_.get(), staging_vertex_buffer->Size());
        CopyBuffer(cmd_list, staging_index_buffer.get(), index_buffer_.get(), staging_index_buffer->Size());
      });

  device_->CreateShaderModule(CompileShader(GetShaderCode("shaders/main.hlsl"), "RayGenMain", "lib_6_3"),
                              &raygen_shader_);
  device_->CreateShaderModule(CompileShader(GetShaderCode("shaders/main.hlsl"), "MissMain", "lib_6_3"), &miss_shader_);
  device_->CreateShaderModule(CompileShader(GetShaderCode("shaders/main.hlsl"), "ClosestHitMain", "lib_6_3"),
                              &hit_shader_);
  device_->CreateBottomLevelAccelerationStructure(vertex_buffer_.get(), index_buffer_.get(), sizeof(glm::vec3),
                                                  command_queue_.get(), fence_.get(), command_allocator_.get(), &blas_);
  device_->CreateTopLevelAccelerationStructure({{blas_.get(), glm::mat4{1.0f}}}, command_queue_.get(), fence_.get(),
                                               command_allocator_.get(), &tlas_);

  device_->CreateBuffer(sizeof(CameraObject), D3D12_HEAP_TYPE_UPLOAD, &camera_object_buffer_);
  CameraObject camera_object{};
  camera_object.screen_to_camera = glm::inverse(
      glm::perspectiveLH(glm::radians(90.0f), (float)swap_chain_->Width() / (float)swap_chain_->Height(), 0.1f, 10.0f));
  camera_object.camera_to_world = glm::inverse(
      glm::lookAtLH(glm::vec3{0.0f, 0.0f, -2.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}));
  std::memcpy(camera_object_buffer_->Map(), &camera_object, sizeof(CameraObject));
  camera_object_buffer_->Unmap();
}

void Application::DestroyPipelineAssets() {
}

}  // namespace D3D12
