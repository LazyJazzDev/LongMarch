#include <charconv>
#include <iostream>
#include <string>

#include "module.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  std::string module;
  bool tui = false;
  int frames = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tui") {
      tui = true;
      continue;
    }
    if (arg == "--list") {
      std::cout << "hello_window\nhello_triangle\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout
          << "[--module hello_window|hello_triangle | --tui] [--backend auto|d3d12|vulkan|metal] [--frames N]\nNo module specified: open the terminal selection menu.\n";
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
  if (tui && !module.empty()) {
    std::cerr << "Use either --module or --tui, not both\n";
    return 1;
  }
  if (module.empty()) {
    std::cout << "\nModule Selection\n"
                 "  1. hello_window    Clear and present a window\n"
                 "  2. hello_triangle  Colored indexed triangle (D3D12/Vulkan)\n";
    for (;;) {
      std::cout << "\nSelect module [1-2 or name], q to quit: " << std::flush;
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
      std::cout << "Invalid selection. Enter a listed number or module name.\n";
    }
  }
  if (module == "hello_window")
    return HelloWindow(api, frames);
  if (module == "hello_triangle")
    return HelloTriangle(api, frames);
  std::cerr << "Unknown module: " << module << '\n';
  return 1;
}
