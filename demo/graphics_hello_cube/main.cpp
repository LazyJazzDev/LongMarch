#include "app.h"

int main() {
  Application app_d3d12{grassland::graphics::BACKEND_API_D3D12};
  Application app_vulkan{grassland::graphics::BACKEND_API_VULKAN};

  // app_d3d12.OnInit();
  app_vulkan.OnInit();

  grassland::Metronome metronome;

  while (app_d3d12.IsAlive() || app_vulkan.IsAlive()) {
    if (app_d3d12.IsAlive()) {
      app_d3d12.OnUpdate();
    }
    if (app_d3d12.IsAlive()) {
      app_d3d12.OnRender();
    }
    if (app_vulkan.IsAlive()) {
      app_vulkan.OnUpdate();
    }
    if (app_vulkan.IsAlive()) {
      app_vulkan.OnRender();
    }

    glfwPollEvents();
    // metronome.Tick(std::chrono::milliseconds(16));
  }

  app_d3d12.OnClose();
  app_vulkan.OnClose();

  return 0;
}
