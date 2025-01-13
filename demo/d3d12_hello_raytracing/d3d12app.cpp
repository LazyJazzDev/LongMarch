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
  static float theta = 0.0f;
  theta += glm::radians(0.1f);

  tlas_->UpdateInstances(
      std::vector<std::pair<grassland::d3d12::AccelerationStructure *, glm::mat4>>{
          {blas_.get(), glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 1.0f, 0.0f})}},
      command_queue_.get(), fence_.get(), command_allocator_.get());
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

  ComPtr<ID3D12GraphicsCommandList4> dxr_command_list;
  command_list_->Handle()->QueryInterface(IID_PPV_ARGS(&dxr_command_list));

  dxr_command_list->SetComputeRootSignature(root_signature_->Handle());
  D3D12_DISPATCH_RAYS_DESC dispatch_desc = {};
  ID3D12DescriptorHeap *descriptor_heaps[] = {descriptor_heap_->Handle()};
  dxr_command_list->SetDescriptorHeaps(1, descriptor_heaps);
  dxr_command_list->SetComputeRootDescriptorTable(0, descriptor_heap_->GPUHandle(0));

  UINT shader_record_size =
      SizeAlignTo(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
  UINT shader_table_size = SizeAlignTo(shader_record_size, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

  dispatch_desc.HitGroupTable.StartAddress = shader_table_->GetHitGroupDeviceAddress();
  dispatch_desc.HitGroupTable.SizeInBytes = shader_table_size;
  dispatch_desc.HitGroupTable.StrideInBytes = shader_record_size;
  dispatch_desc.MissShaderTable.StartAddress = shader_table_->GetMissDeviceAddress();
  dispatch_desc.MissShaderTable.SizeInBytes = shader_table_size;
  dispatch_desc.MissShaderTable.StrideInBytes = shader_record_size;
  dispatch_desc.RayGenerationShaderRecord.StartAddress = shader_table_->GetRayGenDeviceAddress();
  dispatch_desc.RayGenerationShaderRecord.SizeInBytes = shader_table_size;
  dispatch_desc.Width = frame_image_->Width();
  dispatch_desc.Height = frame_image_->Height();
  dispatch_desc.Depth = 1;
  dxr_command_list->SetPipelineState1(ray_tracing_pipeline_->Handle());

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(frame_image_->Handle(), D3D12_RESOURCE_STATE_GENERIC_READ,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  dxr_command_list->DispatchRays(&dispatch_desc);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(swap_chain_->BackBuffer(back_buffer_index),
                                                 D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(frame_image_->Handle(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  CD3DX12_TEXTURE_COPY_LOCATION src(frame_image_->Handle());
  CD3DX12_TEXTURE_COPY_LOCATION dst(swap_chain_->BackBuffer(back_buffer_index));

  command_list_->Handle()->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(frame_image_->Handle(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ);

  command_list_->Handle()->ResourceBarrier(1, &barrier);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(swap_chain_->BackBuffer(back_buffer_index),
                                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);

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

  glfw_window_ = glfwCreateWindow(1920, 1080, "D3D12 Hello Ray Tracing", nullptr, nullptr);

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

  CD3DX12_DESCRIPTOR_RANGE1 descriptor_ranges[3];
  descriptor_ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
  descriptor_ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 1);
  descriptor_ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 2);

  CD3DX12_ROOT_PARAMETER1 root_parameter;
  root_parameter.InitAsDescriptorTable(3, descriptor_ranges, D3D12_SHADER_VISIBILITY_ALL);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.Init_1_1(1, &root_parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

  device_->CreateRootSignature(root_signature_desc, &root_signature_);

  device_->CreateImage(swap_chain_->Width(), swap_chain_->Height(), DXGI_FORMAT_R8G8B8A8_UNORM, &frame_image_);

  device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, &descriptor_heap_);

  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
  srv_desc.RaytracingAccelerationStructure.Location = tlas_->Handle()->GetGPUVirtualAddress();
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  device_->Handle()->CreateShaderResourceView(nullptr, &srv_desc, descriptor_heap_->CPUHandle(0));

  D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
  uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  uav_desc.Format = frame_image_->Format();
  uav_desc.Texture2D.MipSlice = 0;
  uav_desc.Texture2D.PlaneSlice = 0;
  device_->Handle()->CreateUnorderedAccessView(frame_image_->Handle(), nullptr, &uav_desc,
                                               descriptor_heap_->CPUHandle(1));
  D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc{};
  cbv_desc.BufferLocation = camera_object_buffer_->Handle()->GetGPUVirtualAddress();
  cbv_desc.SizeInBytes = SizeAlignTo(camera_object_buffer_->Size(), 256);
  device_->Handle()->CreateConstantBufferView(&cbv_desc, descriptor_heap_->CPUHandle(2));

  device_->CreateRayTracingPipeline(root_signature_.get(), raygen_shader_.get(), miss_shader_.get(), hit_shader_.get(),
                                    &ray_tracing_pipeline_);

  device_->CreateShaderTable(ray_tracing_pipeline_.get(), &shader_table_);
}

void Application::DestroyPipelineAssets() {
}

}  // namespace D3D12
