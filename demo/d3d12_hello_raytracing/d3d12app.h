#pragma once
#include "long_march.h"

namespace D3D12 {

using namespace long_march;
using namespace long_march::d3d12;

struct CameraObject {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
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

  std::unique_ptr<Buffer> camera_object_buffer_;

  std::unique_ptr<AccelerationStructure> blas_;
  std::unique_ptr<AccelerationStructure> tlas_;

  std::unique_ptr<ShaderModule> raygen_shader_;
  std::unique_ptr<ShaderModule> miss_shader_;
  std::unique_ptr<ShaderModule> hit_shader_;

  std::unique_ptr<RootSignature> root_signature_;
  std::unique_ptr<PipelineState> pipeline_state_;

  std::unique_ptr<Image> texture_;
};

}  // namespace D3D12
