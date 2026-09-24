#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include "demo/gol/cells_pattern.h"
#include "demos/DemoSession.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

static void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

static void Settle(DemoSession &session) {
  for (int i = 0; i < 24; ++i) {
    session.Render();
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
}

static std::vector<uint8_t> Pixels(DemoSession &session) {
  auto extent = session.Image()->Extent();
  std::vector<uint8_t> pixels(extent.width * extent.height * 4);
  session.Image()->DownloadData(pixels.data());
  return pixels;
}

int main(int argc, char **argv) {
  try {
    Check(argc == 2, "usage: mobile_games_check <resources>");
    for (auto name : {"gol", "2048"}) {
      DemoSession game(argv[1], name);
      game.Resize(640, 800);
      Settle(game);
      auto before = Pixels(game);
      Check(*std::max_element(before.begin(), before.end()) > 200, "Shared UI produced no visible geometry");
      auto *window = game.Game()->Window();
      Check(window->IsHosted() && window->GetFramebufferSize() == glm::ivec2(640, 800), "Hosted resize failed");
      window->CursorEnterEvent().InvokeCallbacks(true);
      if (std::string(name) == "gol") {
        window->SendPointer(327, 407);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE);
        Settle(game);
        Check(Pixels(game) != before, "Desktop cell interaction did not change the rendered grid");
        auto edited = Pixels(game);
        window->MagnifyEvent().InvokeCallbacks(
            grassland::graphics::MagnifyGesture{1.5, 320, 400, grassland::graphics::MagnifyPhase::kUpdate});
        Settle(game);
        Check(Pixels(game) != edited, "Desktop grid magnification did not update rendering");
        window->SendKey(GLFW_KEY_O, GLFW_PRESS, GLFW_MOD_CONTROL);
        window->SendKey(GLFW_KEY_O, GLFW_RELEASE, GLFW_MOD_CONTROL);
        Settle(game);
        Check(game.Game()->FileRequest() == 1, "Original open action did not reach the native host");
        Check(game.Game()
                  ->CompleteFile((std::filesystem::path(argv[1]) / "Patterns/gosper-glider-gun.cells").string())
                  .empty(),
              "Desktop pattern import failed");
        window->SendKey(GLFW_KEY_S, GLFW_PRESS, GLFW_MOD_CONTROL);
        window->SendKey(GLFW_KEY_S, GLFW_RELEASE, GLFW_MOD_CONTROL);
        Settle(game);
        Check(game.Game()->FileRequest() == 2, "Original save action did not reach the native host");
        auto path =
            std::filesystem::temp_directory_path() /
            ("longmarch-ui-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".cells");
        Check(game.Game()->CompleteFile(path.string()).empty(), "Hosted grid export failed");
        auto saved = LoadCellsPattern(path.string());
        std::filesystem::remove(path);
        Check(saved.width == 40 && saved.height == 30 && std::count(saved.cells.begin(), saved.cells.end(), 1) == 36,
              "Import/export lost centered pattern cells or grid dimensions");
      } else {
        for (int key : {GLFW_KEY_LEFT, GLFW_KEY_UP, GLFW_KEY_RIGHT, GLFW_KEY_DOWN}) {
          window->SendKey(key, GLFW_PRESS);
          window->SendKey(key, GLFW_RELEASE);
          Settle(game);
        }
        Check(Pixels(game) != before, "Desktop puzzle moves did not change the rendered tiles");
      }
      window->SendFocus(false);
      Check(!window->IsFocused() && !window->IsMouseButtonDown(0), "Hosted focus did not release inputs");
      window->SendFocus(true);
      game.Resize(800, 480);
      Settle(game);
      Check(game.Image()->Extent().width == 800 && game.Image()->Extent().height == 480,
            "Shared UI orientation resize failed");
      std::cout << "PASS " << name << " shared desktop renderer, input, animation, focus and orientation\n";
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
