#pragma once
#include "long_march.h"

namespace D3D12 {

using namespace long_march;
using namespace long_march::d3d12;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class Application {
 public:
  Application();
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();

  void CreateWindowAssets();
  void DestroyWindowAssets();

  void CreatePipelineAssets();
  void DestroyPipelineAssets();

  const uint32_t frame_count = 2;

  GLFWwindow *glfw_window_;

  std::unique_ptr<DXGIFactory> factory_;
  std::unique_ptr<Device> device_;

  std::unique_ptr<CommandQueue> command_queue_;
  std::unique_ptr<CommandAllocator> command_allocator_;
  std::unique_ptr<CommandList> command_list_;

  std::unique_ptr<SwapChain> swap_chain_;

  std::unique_ptr<Fence> fence_;

  std::unique_ptr<Buffer> vertex_buffer_;
  std::unique_ptr<Buffer> index_buffer_;

  std::unique_ptr<ShaderModule> vertex_shader_;
  std::unique_ptr<ShaderModule> pixel_shader_;

  std::unique_ptr<RootSignature> root_signature_;
  std::unique_ptr<PipelineState> pipeline_state_;

  std::unique_ptr<Buffer> uniform_buffer_;
  std::unique_ptr<DescriptorHeap> descriptor_heap_;

  std::unique_ptr<Image> texture_;
  std::unique_ptr<Image> depth_image_;
  std::unique_ptr<DescriptorHeap> dsv_descriptor_heap_;
};

}  // namespace D3D12
