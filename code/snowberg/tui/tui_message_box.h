#pragma once

#include <string>
#include <vector>

#include "snowberg/tui/tui_core.h"

namespace snowberg::tui {
class MessageBox : public Core {
 public:
  // Framed dialog with a title and OK button. Enter/Escape/Space or clicking OK closes it.
  // Long messages wrap and can be scrolled.
  // Unavailable terminals print a framed message and wait for Enter instead.
  // Returns false on EOF, I/O error or recursive use. Instances can be reused.
  bool Show(const std::string &title, const std::string &message);

 protected:
  void OnResize(int columns, int rows) override;
  void OnKey(const KeyEvent &event) override;
  void OnMouse(const MouseEvent &event) override;
  void OnUpdate() override;

 private:
  using Core::Run;
  void Scroll(int amount);
  bool showing_{}, acknowledged_{}, dirty_{};
  std::string title_, message_;
  std::vector<std::string> lines_;
  int top_{}, visible_{};
  int left_{}, upper_{}, width_{}, height_{}, text_row_{}, text_column_{}, button_row_{}, button_column_{};
};
}  // namespace snowberg::tui
