#include "app.h"

Application::Application() {
  grassland::graphics::CreateCore(grassland::graphics::BACKEND_API_D3D12,
                                  grassland::graphics::Core::Settings{},
                                  &core_);

  auto d3d12_core =
      std::dynamic_pointer_cast<grassland::graphics::backend::D3D12Core>(core_);
  auto vulkan_core =
      std::dynamic_pointer_cast<grassland::graphics::backend::VulkanCore>(
          core_);

  if (d3d12_core) {
    puts("Is D3D12Core");
  } else {
    puts("Not D3D12Core");
  }

  if (vulkan_core) {
    puts("Is VulkanCore");
  } else {
    puts("Not VulkanCore");
  }
}
