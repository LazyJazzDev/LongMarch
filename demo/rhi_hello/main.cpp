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
      std::cout << "window\ntriangle\ncube\ntexture\nblend\nnbody\nraytracing\nray_query\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout
          << "[--module window|triangle|cube|texture|blend|nbody|raytracing|ray_query | --tui | --all] [--backend auto|d3d12|vulkan|metal] [--headless] [--frames N]\nHeadless defaults to 3000 frames per scene; --all defaults to 300 frames with windows.\nNo module specified: open the terminal selection menu.\n";
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
    if (RunWindow(api, frames, headless) != 0) {
      std::cerr << "window: FAILED\n";
      result = 1;
    }
    if (RunTriangle(api, frames, headless) != 0) {
      std::cerr << "triangle: FAILED\n";
      result = 1;
    }
    if (RunCube(api, frames, headless) != 0) {
      std::cerr << "cube: FAILED\n";
      result = 1;
    }
    if (RunTexture(api, frames, headless) != 0) {
      std::cerr << "texture: FAILED\n";
      result = 1;
    }
    if (RunBlend(api, frames, headless) != 0) {
      std::cerr << "blend: FAILED\n";
      result = 1;
    }
    if (RunNBody(api, frames, headless) != 0) {
      std::cerr << "nbody: FAILED\n";
      result = 1;
    }
    if (RunRayTracing(api, frames, headless) != 0) {
      std::cerr << "raytracing: FAILED\n";
      result = 1;
    }
    if (RunRayQuery(api, frames, headless) != 0) {
      std::cerr << "ray_query: FAILED\n";
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
                 "  1. window    Clear and present a window\n"
                 "  2. triangle  Colored indexed triangle (D3D12/Vulkan/Metal)\n"
                 "  3. cube      Rotating cube with depth and uniform binding\n"
                 "  4. texture   Sampled XOR texture triangle\n"
                 "  5. blend     Source-alpha blended triangles\n"
                 "  6. nbody     Compute simulation and particle rendering (D3D12/Vulkan/Metal)\n";
    std::cout << "  7. raytracing  Ray pipeline, triangles and procedural sphere (D3D12/Vulkan)\n"
                 "  8. ray_query   Inline ray queries (D3D12/Vulkan/Metal)\n";
    for (;;) {
      std::cout << "\nSelect module [1-8 or name], q to quit: " << std::flush;
      std::string input;
      if (!std::getline(std::cin, input))
        return 0;
      const auto first = input.find_first_not_of(" \t\r");
      if (first == std::string::npos)
        continue;
      input = input.substr(first, input.find_last_not_of(" \t\r") - first + 1);
      if (input == "q" || input == "quit")
        return 0;
      if (input == "1" || input == "window") {
        module = "window";
        break;
      }
      if (input == "2" || input == "triangle") {
        module = "triangle";
        break;
      }
      if (input == "3" || input == "cube") {
        module = "cube";
        break;
      }
      if (input == "4" || input == "texture") {
        module = "texture";
        break;
      }
      if (input == "5" || input == "blend") {
        module = "blend";
        break;
      }
      if (input == "6" || input == "nbody") {
        module = "nbody";
        break;
      }
      if (input == "7" || input == "raytracing") {
        module = "raytracing";
        break;
      }
      if (input == "8" || input == "ray_query") {
        module = "ray_query";
        break;
      }
      std::cout << "Invalid selection. Enter a listed number or module name.\n";
    }
  }
  if (module == "window")
    return RunWindow(api, frames, headless);
  if (module == "triangle")
    return RunTriangle(api, frames, headless);
  if (module == "cube")
    return RunCube(api, frames, headless);
  if (module == "texture")
    return RunTexture(api, frames, headless);
  if (module == "blend")
    return RunBlend(api, frames, headless);
  if (module == "nbody")
    return RunNBody(api, frames, headless);
  if (module == "raytracing")
    return RunRayTracing(api, frames, headless);
  if (module == "ray_query")
    return RunRayQuery(api, frames, headless);
  std::cerr << "Unknown module: " << module << '\n';
  return 1;
}
