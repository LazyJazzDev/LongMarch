#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

struct BlitPipeline {
  d3d12::Device *device_;
  std::unique_ptr<d3d12::ShaderModule> vertex_shader;
  std::unique_ptr<d3d12::ShaderModule> pixel_shader;
  std::unique_ptr<d3d12::RootSignature> root_signature;
  std::map<DXGI_FORMAT, std::unique_ptr<d3d12::PipelineState>> pipeline_states;
  void Initialize(d3d12::Device *device);
  d3d12::PipelineState *GetPipelineState(DXGI_FORMAT format);
};

class D3D12Core : public Core {
 public:
  D3D12Core(const Settings &settings);
  ~D3D12Core() override;

  BackendAPI API() const override {
    return BACKEND_API_D3D12;
  }

  int CreateBuffer(size_t size,
                   BufferType type,
                   double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width,
                  int height,
                  ImageFormat format,
                  double_ptr<Image> pp_image) override;

  int CreateWindowObject(int width,
                         int height,
                         const std::string &title,
                         double_ptr<Window> pp_window) override;

  int CreateShader(const void *data,
                   size_t size,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateCommandContext(
      double_ptr<CommandContext> pp_command_context) override;

  int SubmitCommandContext(CommandContext *p_command_context) override;

  int GetPhysicalDeviceProperties(
      PhysicalDeviceProperties *p_physical_device_properties =
          nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  void WaitGPU() override;

  d3d12::DXGIFactory *DXGIFactory() const {
    return dxgi_factory_.get();
  }

  d3d12::Device *Device() const {
    return device_.get();
  }

  d3d12::CommandQueue *CommandQueue() const {
    return command_queue_.get();
  }

  d3d12::CommandList *CommandList() const {
    return command_lists_[current_frame_].get();
  }

  d3d12::CommandAllocator *CommandAllocator() const {
    return command_allocators_[current_frame_].get();
  }

  d3d12::Fence *Fence() const {
    return fences_[current_frame_].get();
  }

  uint32_t CurrentFrame() const {
    return current_frame_;
  }

  void SingleTimeCommand(
      std::function<void(ID3D12GraphicsCommandList *)> command);

  BlitPipeline *BlitPipeline() {
    return &blit_pipeline_;
  }

  d3d12::DescriptorHeap *RTVDescriptorHeap() const {
    return rtv_descriptor_heaps_[current_frame_].get();
  }

  d3d12::DescriptorHeap *DSVDescriptorHeap() const {
    return dsv_descriptor_heaps_[current_frame_].get();
  }

 private:
  std::unique_ptr<d3d12::DXGIFactory> dxgi_factory_;
  std::unique_ptr<d3d12::Device> device_;

  struct BlitPipeline blit_pipeline_;

  std::unique_ptr<d3d12::CommandQueue> command_queue_;
  std::unique_ptr<d3d12::CommandQueue> transfer_command_queue_;
  std::vector<std::unique_ptr<d3d12::CommandAllocator>> command_allocators_;
  std::vector<std::unique_ptr<d3d12::CommandList>> command_lists_;

  std::vector<std::unique_ptr<d3d12::Fence>> fences_;

  std::unique_ptr<d3d12::CommandAllocator> single_time_allocator_;
  std::unique_ptr<d3d12::CommandList> single_time_command_list_;
  std::unique_ptr<d3d12::Fence> single_time_fence_;

  std::unique_ptr<d3d12::CommandAllocator> transfer_allocator_;
  std::unique_ptr<d3d12::CommandList> transfer_command_list_;
  std::unique_ptr<d3d12::Fence> transfer_fence_;

  std::vector<std::unique_ptr<d3d12::DescriptorHeap>>
      resource_descriptor_heaps_;

  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> rtv_descriptor_heaps_;
  std::vector<std::unique_ptr<d3d12::DescriptorHeap>> dsv_descriptor_heaps_;

  uint32_t current_frame_{0};
};

}  // namespace grassland::graphics::backend
