#include "d3d12app.h"

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

  command_list_->Handle()->SetPipelineState(pipeline_state_->Handle());
  command_list_->Handle()->SetGraphicsRootSignature(root_signature_->Handle());

  uint32_t width, height;
  swap_chain_->Handle()->GetSourceSize(&width, &height);
  D3D12_RECT scissor_rect = {0, 0, LONG(width), LONG(height)};
  D3D12_VIEWPORT viewport = {0.0f,          0.0f, FLOAT(width),
                             FLOAT(height), 0.0f, 1.0f};

  command_list_->Handle()->RSSetViewports(1, &viewport);
  command_list_->Handle()->RSSetScissorRects(1, &scissor_rect);

  command_list_->Handle()->OMSetRenderTargets(1, &rtv_handle, FALSE, nullptr);
  command_list_->Handle()->IASetPrimitiveTopology(
      D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  D3D12_VERTEX_BUFFER_VIEW vertex_buffer_view = {};
  vertex_buffer_view.BufferLocation =
      vertex_buffer_->Handle()->GetGPUVirtualAddress();
  vertex_buffer_view.StrideInBytes = sizeof(Vertex);
  vertex_buffer_view.SizeInBytes = vertex_buffer_->Size();

  D3D12_INDEX_BUFFER_VIEW index_buffer_view = {};
  index_buffer_view.BufferLocation =
      index_buffer_->Handle()->GetGPUVirtualAddress();
  index_buffer_view.Format = DXGI_FORMAT_R32_UINT;
  index_buffer_view.SizeInBytes = index_buffer_->Size();

  command_list_->Handle()->IASetVertexBuffers(0, 1, &vertex_buffer_view);
  command_list_->Handle()->IASetIndexBuffer(&index_buffer_view);

  command_list_->Handle()->DrawIndexedInstanced(3, 1, 0, 0, 0);

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
  DestroyPipelineAssets();
  DestroyWindowAssets();
}

void Application::CreateWindowAssets() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  glfw_window_ =
      glfwCreateWindow(800, 600, "D3D12 Hello Triangle", nullptr, nullptr);

  CreateDXGIFactory({}, &factory_);

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

void Application::CreatePipelineAssets() {
  std::vector<Vertex> vertices = {
      {{0.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2};

  std::unique_ptr<Buffer> staging_vertex_buffer;
  std::unique_ptr<Buffer> staging_index_buffer;

  device_->CreateBuffer(vertices.size() * sizeof(Vertex),
                        D3D12_HEAP_TYPE_UPLOAD, &staging_vertex_buffer);
  device_->CreateBuffer(indices.size() * sizeof(uint32_t),
                        D3D12_HEAP_TYPE_UPLOAD, &staging_index_buffer);

  device_->CreateBuffer(vertices.size() * sizeof(Vertex),
                        D3D12_HEAP_TYPE_DEFAULT, &vertex_buffer_);
  device_->CreateBuffer(indices.size() * sizeof(uint32_t),
                        D3D12_HEAP_TYPE_DEFAULT, &index_buffer_);
  std::memcpy(staging_vertex_buffer->Map(), vertices.data(),
              vertices.size() * sizeof(Vertex));
  staging_vertex_buffer->Unmap();
  std::memcpy(staging_index_buffer->Map(), indices.data(),
              indices.size() * sizeof(uint32_t));
  staging_index_buffer->Unmap();

  command_queue_->SingleTimeCommand(
      [&staging_vertex_buffer, &staging_index_buffer,
       this](ID3D12GraphicsCommandList *cmd_list) {
        CopyBuffer(cmd_list, staging_vertex_buffer.get(), vertex_buffer_.get(),
                   staging_vertex_buffer->Size());
        CopyBuffer(cmd_list, staging_index_buffer.get(), index_buffer_.get(),
                   staging_index_buffer->Size());
      });

  device_->CreateShaderModule(
      CompileShader(GetShaderCode("shaders/main.hlsl"), "VSMain", "vs_6_0"),
      &vertex_shader_);
  device_->CreateShaderModule(
      CompileShader(GetShaderCode("shaders/main.hlsl"), "PSMain", "ps_6_0"),
      &pixel_shader_);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.Init_1_1(
      0, nullptr, 0, nullptr,
      D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

  device_->CreateRootSignature(root_signature_desc, &root_signature_);

  std::vector<D3D12_INPUT_ELEMENT_DESC> input_element_descs = {
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
  };

  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_state_desc = {};
  pipeline_state_desc.pRootSignature = root_signature_->Handle();
  pipeline_state_desc.VS = vertex_shader_->Handle();
  pipeline_state_desc.PS = pixel_shader_->Handle();
  pipeline_state_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
  pipeline_state_desc.SampleMask = UINT_MAX;
  pipeline_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
  pipeline_state_desc.DepthStencilState.DepthEnable = FALSE;
  pipeline_state_desc.DepthStencilState.StencilEnable = FALSE;
  pipeline_state_desc.InputLayout = {input_element_descs.data(),
                                     UINT(input_element_descs.size())};
  pipeline_state_desc.PrimitiveTopologyType =
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  pipeline_state_desc.NumRenderTargets = 1;
  pipeline_state_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
  pipeline_state_desc.SampleDesc.Count = 1;

  device_->CreatePipelineState(pipeline_state_desc, &pipeline_state_);
}

void Application::DestroyPipelineAssets() {
}

}  // namespace D3D12
