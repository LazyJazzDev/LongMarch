#include "grassland/graphics/backend/d3d12/d3d12_core.h"

#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"
#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_program.h"
#include "grassland/graphics/backend/d3d12/d3d12_sampler.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

namespace {
#include "built_in_shaders.inl"
}

void BlitPipeline::Initialize(d3d12::Device *device) {
  device_ = device;
  device_->CreateShaderModule(d3d12::CompileShader(GetShaderCode("shaders/d3d12/blit.hlsl"), "VSMain", "vs_6_0"),
                              &vertex_shader);
  device_->CreateShaderModule(d3d12::CompileShader(GetShaderCode("shaders/d3d12/blit.hlsl"), "PSMain", "ps_6_0"),
                              &pixel_shader);

  CD3DX12_DESCRIPTOR_RANGE1 range;
  range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);

  CD3DX12_ROOT_PARAMETER1 root_parameter;
  root_parameter.InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
  CD3DX12_STATIC_SAMPLER_DESC sampler_desc(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.Init_1_1(1, &root_parameter, 1, &sampler_desc,
                               D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
  device_->CreateRootSignature(root_signature_desc, &root_signature);
}

d3d12::PipelineState *BlitPipeline::GetPipelineState(DXGI_FORMAT format) {
  if (pipeline_states.count(format) == 0) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_state_desc = {};
    pipeline_state_desc.pRootSignature = root_signature->Handle();
    pipeline_state_desc.VS = vertex_shader->Handle();
    pipeline_state_desc.PS = pixel_shader->Handle();
    pipeline_state_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    pipeline_state_desc.SampleMask = UINT_MAX;
    pipeline_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    pipeline_state_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipeline_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    pipeline_state_desc.DepthStencilState.DepthEnable = FALSE;
    pipeline_state_desc.DepthStencilState.StencilEnable = FALSE;
    pipeline_state_desc.InputLayout = {nullptr, 0};
    pipeline_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_state_desc.NumRenderTargets = 1;
    pipeline_state_desc.RTVFormats[0] = format;
    pipeline_state_desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    pipeline_state_desc.SampleDesc.Count = 1;
    pipeline_state_desc.SampleDesc.Quality = 0;
    device_->CreatePipelineState(pipeline_state_desc, &pipeline_states[format]);
  }
  return pipeline_states.at(format).get();
}

D3D12Core::D3D12Core(const Settings &settings) : Core(settings) {
  d3d12::DXGIFactoryCreateHint hint{DebugEnabled()};
  d3d12::CreateDXGIFactory(hint, &dxgi_factory_);
}

D3D12Core::~D3D12Core() {
}

int D3D12Core::CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) {
  switch (type) {
    case BUFFER_TYPE_DYNAMIC:
      pp_buffer.construct<D3D12DynamicBuffer>(this, size);
      break;
    default:
      pp_buffer.construct<D3D12StaticBuffer>(this, size);
      break;
  }
  return 0;
}

int D3D12Core::CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) {
  pp_image.construct<D3D12Image>(this, width, height, format);
  return 0;
}

int D3D12Core::CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) {
  pp_sampler.construct<D3D12Sampler>(this, info);
  return 0;
}

int D3D12Core::CreateWindowObject(int width,
                                  int height,
                                  const std::string &title,
                                  bool fullscreen,
                                  bool resizable,
                                  double_ptr<Window> pp_window) {
  pp_window.construct<D3D12Window>(this, width, height, title, fullscreen, resizable);
  return 0;
}

int D3D12Core::CreateShader(const std::string &source_code,
                            const std::string &entry_point,
                            const std::string &target,
                            double_ptr<Shader> pp_shader) {
  pp_shader.construct<D3D12Shader>(this, CompileShader(source_code, entry_point, target));
  return 0;
}

int D3D12Core::CreateProgram(const std::vector<ImageFormat> &color_formats,
                             ImageFormat depth_format,
                             double_ptr<Program> pp_program) {
  pp_program.construct<D3D12Program>(this, color_formats, depth_format);
  return 0;
}

int D3D12Core::CreateCommandContext(double_ptr<CommandContext> pp_command_context) {
  pp_command_context.construct<D3D12CommandContext>(this);
  return 0;
}

int D3D12Core::CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                      Buffer *index_buffer,
                                                      uint32_t stride,
                                                      double_ptr<AccelerationStructure> pp_blas) {
  D3D12Buffer *d3d12_vertex_buffer = dynamic_cast<D3D12Buffer *>(vertex_buffer);
  D3D12Buffer *d3d12_index_buffer = dynamic_cast<D3D12Buffer *>(index_buffer);

  assert(d3d12_vertex_buffer != nullptr);
  assert(d3d12_index_buffer != nullptr);

  std::unique_ptr<d3d12::AccelerationStructure> blas;
  device_->CreateBottomLevelAccelerationStructure(d3d12_vertex_buffer->InstantBuffer(),
                                                  d3d12_index_buffer->InstantBuffer(), stride, command_queue_.get(),
                                                  single_time_fence_.get(), single_time_allocator_.get(), &blas);

  pp_blas.construct<D3D12AccelerationStructure>(this, std::move(blas));

  return 0;
}

int D3D12Core::CreateTopLevelAccelerationStructure(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
    double_ptr<AccelerationStructure> pp_tlas) {
  std::vector<std::pair<d3d12::AccelerationStructure *, glm::mat4>> d3d12_objects;
  d3d12_objects.reserve(objects.size());
  for (int i = 0; i < objects.size(); ++i) {
    D3D12AccelerationStructure *d3d12_as = dynamic_cast<D3D12AccelerationStructure *>(objects[i].first);
    assert(d3d12_as != nullptr);
    d3d12_objects.push_back({d3d12_as->Handle(), objects[i].second});
  }

  std::unique_ptr<d3d12::AccelerationStructure> tlas;
  device_->CreateTopLevelAccelerationStructure(d3d12_objects, command_queue_.get(), single_time_fence_.get(),
                                               single_time_allocator_.get(), &tlas);

  pp_tlas.construct<D3D12AccelerationStructure>(this, std::move(tlas));

  return 0;
}

int D3D12Core::SubmitCommandContext(CommandContext *p_command_context) {
  D3D12CommandContext *command_context = dynamic_cast<D3D12CommandContext *>(p_command_context);

  if (command_context->dynamic_buffers_.size()) {
    transfer_allocator_->ResetCommandRecord(transfer_command_list_.get());
    for (auto buffer : command_context->dynamic_buffers_) {
      buffer->TransferData(transfer_command_list_->Handle());
    }
    transfer_command_list_->Handle()->Close();

    ID3D12CommandList *command_lists[] = {transfer_command_list_->Handle()};
    transfer_command_queue_->Handle()->ExecuteCommandLists(1, command_lists);
    transfer_fence_->Signal(transfer_command_queue_.get());
  }

  command_allocators_[current_frame_]->ResetCommandRecord(command_lists_[current_frame_].get());

  auto command_list = command_lists_[current_frame_]->Handle();

  for (auto window : command_context->windows_) {
    command_context->RecordRTVImage(window->CurrentBackBuffer());
  }

  if (!resource_descriptor_heaps_[current_frame_] ||
      resource_descriptor_heaps_[current_frame_]->NumDescriptors() < command_context->resource_descriptor_count_) {
    resource_descriptor_heaps_[current_frame_].reset();
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, command_context->resource_descriptor_count_,
                                  &resource_descriptor_heaps_[current_frame_]);
  }

  if (!sampler_descriptor_heaps_[current_frame_] ||
      sampler_descriptor_heaps_[current_frame_]->NumDescriptors() < command_context->sampler_descriptor_count_) {
    sampler_descriptor_heaps_[current_frame_].reset();
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, command_context->sampler_descriptor_count_,
                                  &sampler_descriptor_heaps_[current_frame_]);
  }

  if (command_context->resource_descriptor_count_) {
    command_context->resource_descriptor_size_ = resource_descriptor_heaps_[current_frame_]->DescriptorSize();
    command_context->resource_descriptor_base_ = resource_descriptor_heaps_[current_frame_]->CPUHandle(0);
    command_context->resource_descriptor_gpu_base_ = resource_descriptor_heaps_[current_frame_]->GPUHandle(0);
  }

  if (command_context->sampler_descriptor_count_) {
    command_context->sampler_descriptor_size_ = sampler_descriptor_heaps_[current_frame_]->DescriptorSize();
    command_context->sampler_descriptor_base_ = sampler_descriptor_heaps_[current_frame_]->CPUHandle(0);
    command_context->sampler_descriptor_gpu_base_ = sampler_descriptor_heaps_[current_frame_]->GPUHandle(0);
  }

  if (!rtv_descriptor_heaps_[current_frame_] ||
      rtv_descriptor_heaps_[current_frame_]->NumDescriptors() < command_context->rtv_index_.size()) {
    rtv_descriptor_heaps_[current_frame_].reset();
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, command_context->rtv_index_.size(),
                                  &rtv_descriptor_heaps_[current_frame_]);
  }

  if (!dsv_descriptor_heaps_[current_frame_] ||
      dsv_descriptor_heaps_[current_frame_]->NumDescriptors() < command_context->dsv_index_.size()) {
    dsv_descriptor_heaps_[current_frame_].reset();
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, command_context->dsv_index_.size(),
                                  &dsv_descriptor_heaps_[current_frame_]);
  }

  for (auto &[resource, index] : command_context->rtv_index_) {
    auto rtv_handle = rtv_descriptor_heaps_[current_frame_]->CPUHandle(index);
    device_->Handle()->CreateRenderTargetView(resource, nullptr, rtv_handle);
  }

  for (auto &[resource, index] : command_context->dsv_index_) {
    auto dsv_handle = dsv_descriptor_heaps_[current_frame_]->CPUHandle(index);
    device_->Handle()->CreateDepthStencilView(resource, nullptr, dsv_handle);
  }

  ID3D12DescriptorHeap *resource_heaps[] = {resource_descriptor_heaps_[current_frame_]->Handle(),
                                            sampler_descriptor_heaps_[current_frame_]->Handle()};
  command_list->SetDescriptorHeaps(2, resource_heaps);

  for (auto &command : command_context->commands_) {
    command->CompileCommand(command_context, command_list);
  }

  for (auto &[image, state] : command_context->resource_states_) {
    if (state != D3D12_RESOURCE_STATE_GENERIC_READ) {
      CD3DX12_RESOURCE_BARRIER barrier =
          CD3DX12_RESOURCE_BARRIER::Transition(image, state, D3D12_RESOURCE_STATE_GENERIC_READ);
      command_list->ResourceBarrier(1, &barrier);
    }
  }

  command_list->Close();

  command_context->dsv_index_.clear();
  command_context->rtv_index_.clear();

  ID3D12CommandList *command_lists[] = {command_list};
  command_queue_->Handle()->ExecuteCommandLists(1, command_lists);

  for (auto window : command_context->windows_) {
    window->SwapChain()->Handle()->Present(0, 0);
  }

  fences_[current_frame_]->Signal(command_queue_.get());

  current_frame_ = (current_frame_ + 1) % FramesInFlight();
  fences_[current_frame_]->Wait();
  transfer_fence_->Wait();

  return 0;
}

int D3D12Core::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties) {
  auto adapters = dxgi_factory_->EnumerateAdapters();
  if (adapters.empty()) {
    return 0;
  }

  if (p_physical_device_properties) {
    for (int i = 0; i < adapters.size(); ++i) {
      auto adapter = adapters[i];
      PhysicalDeviceProperties properties{};
      properties.name = adapter.Name();
      properties.score = adapter.Evaluate();
      properties.ray_tracing_support = adapter.SupportRayTracing();
      p_physical_device_properties[i] = properties;
    }
  }

  return adapters.size();
}

int D3D12Core::InitializeLogicalDevice(int device_index) {
  auto adapters = dxgi_factory_->EnumerateAdapters();

  if (device_index < 0 || device_index >= adapters.size()) {
    return -1;
  }

  dxgi_factory_->CreateDevice(d3d12::DeviceFeatureRequirement{adapters[device_index].SupportRayTracing()}, device_index,
                              &device_);

  device_name_ = adapters[device_index].Name();
  ray_tracing_support_ = adapters[device_index].SupportRayTracing();

  device_->CreateCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_queue_);
  command_allocators_.resize(FramesInFlight());
  command_lists_.resize(FramesInFlight());
  fences_.resize(FramesInFlight());
  resource_descriptor_heaps_.resize(FramesInFlight());
  sampler_descriptor_heaps_.resize(FramesInFlight());
  rtv_descriptor_heaps_.resize(FramesInFlight());
  dsv_descriptor_heaps_.resize(FramesInFlight());

  for (int i = 0; i < FramesInFlight(); i++) {
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64, &resource_descriptor_heaps_[i]);
    device_->CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 64, &sampler_descriptor_heaps_[i]);

    device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_allocators_[i]);
    device_->CreateFence(&fences_[i]);

    command_allocators_[i]->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_lists_[i]);
  }

  device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, &single_time_allocator_);
  device_->CreateFence(&single_time_fence_);
  single_time_allocator_->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, &single_time_command_list_);

  device_->CreateCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, &transfer_command_queue_);
  device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, &transfer_allocator_);
  device_->CreateFence(&transfer_fence_);
  transfer_allocator_->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, &transfer_command_list_);

  blit_pipeline_.Initialize(device_.get());

  return 0;
}

void D3D12Core::WaitGPU() {
  single_time_fence_->Signal(command_queue_.get());
  single_time_fence_->Wait();
}

void D3D12Core::SingleTimeCommand(std::function<void(ID3D12GraphicsCommandList *)> command) {
  single_time_allocator_->ResetCommandRecord(single_time_command_list_.get());
  command(single_time_command_list_->Handle());
  single_time_command_list_->Handle()->Close();

  ID3D12CommandList *command_lists[] = {single_time_command_list_->Handle()};
  command_queue_->Handle()->ExecuteCommandLists(1, command_lists);
  single_time_fence_->Signal(command_queue_.get());
  single_time_fence_->Wait();
}

}  // namespace grassland::graphics::backend
