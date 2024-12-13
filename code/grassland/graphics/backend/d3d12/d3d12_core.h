#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

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

 private:
  std::unique_ptr<d3d12::DXGIFactory> dxgi_factory_;
  std::unique_ptr<d3d12::Device> device_;

  std::unique_ptr<d3d12::CommandQueue> command_queue_;
  std::vector<std::unique_ptr<d3d12::CommandAllocator>> command_allocators_;
  std::vector<std::unique_ptr<d3d12::CommandList>> command_lists_;

  std::vector<std::unique_ptr<d3d12::Fence>> fences_;

  std::unique_ptr<d3d12::CommandAllocator> single_time_allocator_;
  std::unique_ptr<d3d12::CommandList> single_time_command_list_;
  std::unique_ptr<d3d12::Fence> single_time_fence_;

  uint32_t current_frame_{0};
};

}  // namespace grassland::graphics::backend
