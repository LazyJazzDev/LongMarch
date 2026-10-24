#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "snowberg/tui/tui_core.h"

namespace snowberg::tui {
struct MenuOption {
  std::string name;
  std::string description;
};

class Menu : public Core {
 public:
  // Blocks until confirmation/cancellation. Returns a zero-based index or
  // nullopt on cancellation, failure, empty options, or recursive selection.
  // Unavailable terminals use a sequential number/name prompt. An instance
  // can serve successive selections; each call resets its menu state.
  std::optional<size_t> Select(const std::string &title,
                               const std::vector<MenuOption> &options,
                               size_t initial_selection = 0);

 protected:
  void OnResize(int columns, int rows) override;
  void OnKey(const KeyEvent &event) override;
  void OnMouse(const MouseEvent &event) override;
  void OnUpdate() override;

 private:
  // Select configures the menu before entering Core's loop.
  using Core::Run;
  using Clock = std::chrono::steady_clock;
  void Layout();
  void SelectIndex(size_t index);
  bool selecting_{};
  std::string title_;
  std::vector<std::string> labels_;
  size_t selected_{}, top_{}, clicked_{};
  int first_row_{}, count_{};
  Clock::time_point selected_at_{}, clicked_at_{};
  std::vector<std::pair<std::string, bool>> previous_;
  std::optional<size_t> result_;
};
}  // namespace snowberg::tui
