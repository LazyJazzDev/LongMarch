#pragma once
#include "long_march.h"

namespace D3D12 {

using namespace long_march;
using namespace long_march::d3d12;

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

  GLFWwindow *glfw_window_;

  std::unique_ptr<grassland::d3d12::DXGIFactory> factory_;
};

}  // namespace D3D12
