#include "snowberg/tui/tui_message_box.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace snowberg::tui {
namespace {
std::vector<std::string> WrapLines(const std::string &text, int width) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty())
      lines.emplace_back();
    for (size_t offset = 0; offset < line.size();) {
      const auto fitted = detail::FitRow(line.substr(offset), width);
      if (fitted.bytes == 0)
        break;
      lines.push_back(fitted.text);
      offset += fitted.bytes;
    }
  }
  return lines;
}

bool ShowText(const std::string &title, const std::string &message) {
  constexpr int width = 72;
  const std::string border = "+" + std::string(width + 2, '-') + "+\n";
  const auto print_lines = [&](const std::string &text) {
    for (const auto &line : WrapLines(text, width))
      std::cout << "| " << line << std::string(width - detail::TextWidth(line), ' ') << " |\n";
  };
  std::cout << '\n' << border;
  print_lines(title);
  std::cout << border;
  print_lines(message);
  std::cout << border << "Press Enter to continue..." << std::flush;
  if (!std::cout)
    return false;
  std::string line;
  return static_cast<bool>(std::getline(std::cin, line));
}
}  // namespace

bool MessageBox::Show(const std::string &title, const std::string &message) {
  if (showing_)
    return false;
  showing_ = true;

  struct ResetShowing {
    bool &showing;

    ~ResetShowing() {
      showing = false;
    }
  } reset{showing_};

  title_ = title;
  message_ = message;
  top_ = 0;
  acknowledged_ = false;
  dirty_ = true;
  const auto status = Run();
  if (status == RunResult::Unavailable)
    return ShowText(title, message);
  return status == RunResult::Completed && acknowledged_;
}

void MessageBox::OnResize(int columns, int rows) {
  lines_ = WrapLines(message_, std::max(1, columns - 1));
  visible_ = std::max(0, rows - (rows > 2 ? 2 : 1));
  top_ = std::min(top_, std::max(0, int(lines_.size()) - visible_));
  dirty_ = true;
}

void MessageBox::Scroll(int amount) {
  top_ = std::clamp(top_ + amount, 0, std::max(0, int(lines_.size()) - visible_));
  dirty_ = true;
}

void MessageBox::OnKey(const KeyEvent &event) {
  if (!event.pressed)
    return;
  if (event.key == Key::Enter || event.key == Key::Escape || (event.key == Key::Character && event.character == U' ')) {
    acknowledged_ = true;
    Terminate();
  } else if (event.key == Key::Up) {
    Scroll(-1);
  } else if (event.key == Key::Down) {
    Scroll(1);
  } else if (event.key == Key::PageUp) {
    Scroll(-visible_);
  } else if (event.key == Key::PageDown) {
    Scroll(visible_);
  }
}

void MessageBox::OnMouse(const MouseEvent &event) {
  if (event.action == MouseAction::Scroll)
    Scroll(-event.scroll_y);
}

void MessageBox::OnUpdate() {
  if (!dirty_)
    return;
  dirty_ = false;
  for (int row = 0; row < Rows(); ++row)
    ClearRow(row);
  const int first_row = Rows() > 2 ? 1 : 0;
  if (first_row) {
    SetInverse(true);
    UpdateRow(0, 0, title_);
    SetInverse(false);
  }
  for (int row = 0; row < visible_ && top_ + row < int(lines_.size()); ++row)
    UpdateRow(row + first_row, 0, lines_[top_ + row]);
  SetForegroundColor(235, 215, 140);
  UpdateRow(Rows() - 1, 0, "Enter / Esc / Space: continue | Up / Down: scroll");
  ResetColors();
}
}  // namespace snowberg::tui
