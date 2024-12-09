#include "app.h"

int main() {
  Application app_d3d12{grassland::graphics::BACKEND_API_D3D12};
  Application app_vulkan{grassland::graphics::BACKEND_API_VULKAN};
  return 0;
}
