#include <chrono>
#include <iostream>

#include "demo/gol/game_of_life_gui.h"
#include "demos/DemoSession.h"

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  // Set the same resource lookup and offline shader cache as the app.
  DemoSession resources(argv[1], "gol");
  for (int size : {40, 200}) {
    life_demo::GameOfLife game("benchmark", 1290, 2409, size, size, grassland::graphics::BACKEND_API_METAL, true);
    game.SetRandomInitialCells(.3f, 42);
    game.SetInitialPlaying(true);
    game.InitializeHosted();
    const auto input_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i)
      game.GetWindow()->SendPointer(600 + i % 10, 1200);
    const double input_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - input_start).count() / 100;
    double cpu = 0, total = 0;
    for (int i = 0; i < 70; ++i) {
      const auto start = std::chrono::steady_clock::now();
      game.RenderHostedFrame();
      const auto submitted = std::chrono::steady_clock::now();
      game.Core()->WaitGPU();
      if (i >= 10) {
        cpu += std::chrono::duration<double, std::milli>(submitted - start).count();
        total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      }
    }
    std::cout << size << "x" << size << " CPU/submit " << cpu / 60 << " ms, frame " << total / 60 << " ms, pointer "
              << input_ms << " ms\n";
    game.CloseHosted();
  }
}
