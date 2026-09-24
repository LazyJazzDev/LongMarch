#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cells_pattern.h"
#include "game_of_life_gui.h"

namespace {

graphics::BackendAPI ParseBackend(const std::string &name) {
  if (name == "auto") {
    return graphics::BACKEND_API_DEFAULT;
  }
  if (name == "vulkan") {
    return graphics::BACKEND_API_VULKAN;
  }
  if (name == "d3d12") {
    return graphics::BACKEND_API_D3D12;
  }
  if (name == "metal") {
    return graphics::BACKEND_API_METAL;
  }
  throw std::invalid_argument("Unknown backend: " + name);
}

void PrintHelp(const char *executable) {
  std::cout << "Usage: " << executable << " [WIDTH HEIGHT] [--backend auto|vulkan|d3d12|metal]\n"
            << "       [--random DENSITY | --pattern FILE] [--play] [--frames N] [--screenshot FILE]\n"
            << "  WIDTH HEIGHT       Cell grid size, each in [2, 200] (default 40 30)\n"
            << "  --random DENSITY   Start with random live cells, e.g. 0.3\n"
            << "  --pattern FILE     Center a Life .cells pattern in the grid\n"
            << "  --play             Start the simulation immediately\n"
            << "  --frames N         Exit after N rendered frames\n"
            << "  --screenshot FILE  Save the last frame as PNG on exit\n";
}

}  // namespace

int main(int argc, char *argv[]) {
  try {
    int cell_grid_width = 40;
    int cell_grid_height = 30;
    int frames = 0;
    auto api = graphics::BACKEND_API_DEFAULT;
    std::string screenshot;
    float random_density = 0.0f;
    bool random_requested = false;
    std::string pattern_path;
    bool play = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
      const std::string option = argv[i];
      if (option == "--help") {
        PrintHelp(argv[0]);
        return 0;
      } else if (option == "--backend" && i + 1 < argc) {
        api = ParseBackend(argv[++i]);
      } else if (option == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
      } else if (option == "--random" && i + 1 < argc) {
        random_density = std::stof(argv[++i]);
        random_requested = true;
      } else if (option == "--pattern" && i + 1 < argc) {
        pattern_path = argv[++i];
      } else if (option == "--play") {
        play = true;
      } else if (option == "--screenshot" && i + 1 < argc) {
        screenshot = argv[++i];
      } else if (!option.empty() && option[0] != '-') {
        positional.push_back(option);
      } else {
        throw std::invalid_argument("Unknown or incomplete option: " + option);
      }
    }

    if (positional.size() == 2) {
      cell_grid_width = std::stoi(positional[0]);
      cell_grid_height = std::stoi(positional[1]);
    } else if (!positional.empty()) {
      throw std::invalid_argument("Expected both WIDTH and HEIGHT for the cell grid");
    }

    // Report error if cell grid size is not in [2,200]*[2,200]
    if (cell_grid_width < 2 || cell_grid_width > grid_size::kMax || cell_grid_height < 2 ||
        cell_grid_height > grid_size::kMax) {
      LogError("Cell grid size must be in [2,200]*[2,200]");
      return 1;
    }
    if (!pattern_path.empty() && random_requested)
      throw std::invalid_argument("Use either --pattern or --random");

    std::vector<uint8_t> initial_cells;
    if (!pattern_path.empty())
      initial_cells = CenterCellsPattern(LoadCellsPattern(pattern_path), cell_grid_width, cell_grid_height);

    GameOfLife app("Game of Life", 1280, 720, cell_grid_width, cell_grid_height, api);
    app.SetInitialPlaying(play);
    if (!pattern_path.empty()) {
      app.SetInitialCells(std::move(initial_cells));
    } else {
      app.SetRandomInitialCells(random_density, 0);
    }
    app.SetScreenshotPath(screenshot);
    app.Run(frames);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
