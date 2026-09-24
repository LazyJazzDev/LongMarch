#include <algorithm>
#include <chrono>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <thread>

#include "demo/gol/application/listener.h"
#include "demo/gol/cells_pattern.h"
#include "demo/gol/simulation_clock.h"
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

static void CheckListenerMutation() {
  life_demo::Application app("listener check", 32, 32, grassland::graphics::BACKEND_API_METAL, true);

  struct Probe : life_demo::Listener {
    explicit Probe(life_demo::Application *app) : Listener(app) {
    }

    std::function<void()> action;
    int calls{};

    void OnCursorPos(double, double) override {
      ++calls;
      if (action)
        action();
    }
  } a(&app), b(&app);

  auto first = std::less<life_demo::Listener *>{}(&a, &b) ? &a : &b;
  auto second = first == &a ? &b : &a;
  first->action = [&] { app.UnregisterListener(second); };
  app.GetWindow()->SendPointer(1, 1);
  Check(first->calls == 1 && second->calls == 0, "Removed listener received a stale snapshot event");
  first->action = {};
  app.RegisterListener(second);
  app.GetWindow()->SendPointer(2, 2);
  Check(first->calls == 2 && second->calls == 1, "Re-registered listener missed input");
  app.UnregisterListener(&a);
  app.UnregisterListener(&b);
}

int main(int argc, char **argv) {
  try {
    Check(argc == 2, "usage: mobile_games_check <resources>");
    CheckListenerMutation();
    SimulationClock clock;
    int steps = 0;
    clock.Advance(.2, true, 0, [&] { ++steps; });
    Check(std::abs(clock.NextStepDelay(0) - .3) < 1e-6, "Slow Life wake deadline drifted");
    clock.Advance(.3, true, 0, [&] { ++steps; });
    Check(steps == 1 && clock.NextStepDelay(0) == .5, "Timed Life wake missed a generation");
    clock.Advance(10, true, SimulationClock::kLightning, [&] { ++steps; });
    Check(steps == 2 && clock.NextStepDelay(SimulationClock::kLightning) == 0,
          "Lightning mode must advance only once per rendered frame");
    for (auto name : {"gol", "2048"}) {
      DemoSession game(argv[1], name);
      game.Resize(640, 800);
      Settle(game);
      Check(std::isinf(game.Game()->NextFrameDelay()), "Settled game still requests continuous frames");
      auto before = Pixels(game);
      Check(*std::max_element(before.begin(), before.end()) > 200, "Shared UI produced no visible geometry");
      auto *window = game.Game()->Window();
      Check(window->IsHosted() && window->GetFramebufferSize() == glm::ivec2(640, 800), "Hosted resize failed");
      window->CursorEnterEvent().InvokeCallbacks(true);
      if (std::string(name) == "gol") {
        game.Game()->SetIconOrientation(glm::half_pi<float>());
        Check(game.Game()->NextFrameDelay() == 0, "Orientation failed to wake paused Life");
        Settle(game);
        const auto rotated = Pixels(game);
        Check(rotated != before, "Orientation did not rotate button icons");
        for (int y = 0; y < 800; ++y)
          for (int x = 128; x < 512; ++x)
            for (int c = 0; c < 4; ++c)
              Check(rotated[(y * 640 + x) * 4 + c] == before[(y * 640 + x) * 4 + c],
                    "Icon rotation changed the grid or its background");
        Check(std::isinf(game.Game()->NextFrameDelay()), "Orientation animation did not return to idle");
        game.Game()->SetIconOrientation(0);
        Settle(game);
        window->SendPointer(327, 407);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE);
        game.Game()->ResetClock();
        game.Render();
        Check(game.Game()->NextFrameDelay() == 0, "Cell edit failed to wake its animation");
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
          game.Game()->ResetClock();
          game.Render();
          Check(game.Game()->NextFrameDelay() == 0, "Puzzle move failed to request animation frames");
          Settle(game);
        }
        Check(Pixels(game) != before, "Desktop puzzle moves did not change the rendered tiles");
        auto board = Pixels(game);
        window->SendPointer(488, 124);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS);
        window->SendMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE);
        game.Game()->ResetClock();
        game.Render();
        Check(game.Game()->NextFrameDelay() == 0, "Menu transition did not wake rendering");
        Settle(game);
        Settle(game);
        Check(Pixels(game) != board && std::isinf(game.Game()->NextFrameDelay()),
              "Lazy overlay target failed to render or settle");
      }
      // File feedback can last over a second; all animations must eventually sleep.
      for (int i = 0; i < 150 && std::isfinite(game.Game()->NextFrameDelay()); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        game.Render();
      }
      Check(std::isinf(game.Game()->NextFrameDelay()), "Game did not sleep after interaction");
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
