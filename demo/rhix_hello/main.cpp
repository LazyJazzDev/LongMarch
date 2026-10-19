#include <charconv>
#include <iostream>
#include <string>

#include "module.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  std::string module = "hello_window";
  int frames = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--list") {
      std::cout << "hello_window\nhello_triangle\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout << "[--module hello_window|hello_triangle] [--backend d3d12|vulkan|metal] [--frames N]\n";
      return 0;
    }
    if (arg == "--module" && i + 1 < argc) {
      module = argv[++i];
      continue;
    }
    if (arg == "--frames" && i + 1 < argc) {
      std::string value = argv[++i];
      auto result = std::from_chars(value.data(), value.data() + value.size(), frames);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || frames < 0)
        return 1;
      continue;
    }
    if (arg == "--backend" && i + 1 < argc)
      arg = argv[++i];
    if (arg == "d3d12")
      api = BACKEND_API_D3D12;
    else if (arg == "vulkan")
      api = BACKEND_API_VULKAN;
    else if (arg == "metal")
      api = BACKEND_API_METAL;
    else {
      std::cerr << "Unknown option: " << arg << '\n';
      return 1;
    }
  }
  if (module == "hello_window")
    return HelloWindow(api, frames);
  if (module == "hello_triangle")
    return HelloTriangle(api, frames);
  std::cerr << "Unknown module: " << module << '\n';
  return 1;
}
