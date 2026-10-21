#include <charconv>
#include <iostream>
#include <string>

#include "module.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  std::string module;
  bool tui = false;
  bool headless = false;
  bool all = false;
  int frames = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--headless") {
      headless = true;
      continue;
    }
    if (arg == "--all") {
      all = true;
      continue;
    }
    if (arg == "--tui") {
      tui = true;
      continue;
    }
    if (arg == "--list") {
      std::cout << "hello_window\nhello_triangle\nhello_cube\nhello_texture\nhello_blend\nhello_nbody\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout
          << "[--module hello_window|hello_triangle|hello_cube|hello_texture|hello_blend|hello_nbody | --tui | --all] [--backend auto|d3d12|vulkan|metal] [--headless] [--frames N]\nHeadless defaults to 3000 frames per scene; --all defaults to 300 frames with windows.\nNo module specified: open the terminal selection menu.\n";
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
    if (arg == "auto")
      api = BACKEND_API_DEFAULT;
    else if (arg == "d3d12")
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
  if (all && (tui || !module.empty())) {
    std::cerr << "Use --all without --module or --tui\n";
    return 1;
  }
  if (!frames) {
    if (headless)
      frames = 3000;
    else if (all)
      frames = 300;
  }
  if (all) {
    int result = 0;
    if (HelloWindow(api, frames, headless) != 0) {
      std::cerr << "hello_window: FAILED\n";
      result = 1;
    }
    if (HelloTriangle(api, frames, headless) != 0) {
      std::cerr << "hello_triangle: FAILED\n";
      result = 1;
    }
    if (HelloCube(api, frames, headless) != 0) {
      std::cerr << "hello_cube: FAILED\n";
      result = 1;
    }
    if (HelloTexture(api, frames, headless) != 0) {
      std::cerr << "hello_texture: FAILED\n";
      result = 1;
    }
    if (HelloBlend(api, frames, headless) != 0) {
      std::cerr << "hello_blend: FAILED\n";
      result = 1;
    }
    if (HelloNBody(api, frames, headless) != 0) {
      std::cerr << "hello_nbody: FAILED\n";
      result = 1;
    }
    return result;
  }
  if (tui && !module.empty()) {
    std::cerr << "Use either --module or --tui, not both\n";
    return 1;
  }
  if (module.empty()) {
    std::cout << "\nModule Selection\n"
                 "  1. hello_window    Clear and present a window\n"
                 "  2. hello_triangle  Colored indexed triangle (D3D12/Vulkan/Metal)\n"
                 "  3. hello_cube      Rotating cube with depth and uniform binding\n"
                 "  4. hello_texture   Sampled XOR texture triangle\n"
                 "  5. hello_blend     Source-alpha blended triangles\n"
                 "  6. hello_nbody     Compute simulation and particle rendering (D3D12/Vulkan)\n";
    for (;;) {
      std::cout << "\nSelect module [1-6 or name], q to quit: " << std::flush;
      std::string input;
      if (!std::getline(std::cin, input))
        return 0;
      const auto first = input.find_first_not_of(" \t\r");
      if (first == std::string::npos)
        continue;
      input = input.substr(first, input.find_last_not_of(" \t\r") - first + 1);
      if (input == "q" || input == "quit")
        return 0;
      if (input == "1" || input == "hello_window") {
        module = "hello_window";
        break;
      }
      if (input == "2" || input == "hello_triangle") {
        module = "hello_triangle";
        break;
      }
      if (input == "3" || input == "hello_cube") {
        module = "hello_cube";
        break;
      }
      if (input == "4" || input == "hello_texture") {
        module = "hello_texture";
        break;
      }
      if (input == "5" || input == "hello_blend") {
        module = "hello_blend";
        break;
      }
      if (input == "6" || input == "hello_nbody") {
        module = "hello_nbody";
        break;
      }
      std::cout << "Invalid selection. Enter a listed number or module name.\n";
    }
  }
  if (module == "hello_window")
    return HelloWindow(api, frames, headless);
  if (module == "hello_triangle")
    return HelloTriangle(api, frames, headless);
  if (module == "hello_cube")
    return HelloCube(api, frames, headless);
  if (module == "hello_texture")
    return HelloTexture(api, frames, headless);
  if (module == "hello_blend")
    return HelloBlend(api, frames, headless);
  if (module == "hello_nbody")
    return HelloNBody(api, frames, headless);
  std::cerr << "Unknown module: " << module << '\n';
  return 1;
}
