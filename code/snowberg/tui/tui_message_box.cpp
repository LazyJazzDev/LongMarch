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
  width_ = std::min(72, columns >= 16 ? columns - 4 : columns);
  left_ = (columns - width_) / 2;
  text_column_ = left_ + (width_ >= 10 ? 3 : 1);
  const int text_width = std::max(1, width_ - 2 * (text_column_ - left_));
  lines_ = WrapLines(message_, text_width);
  const int available_height = rows >= 10 ? rows - 2 : rows;
  height_ = std::min(available_height, std::max(1, int(lines_.size())) + 7);
  upper_ = (rows - height_) / 2;
  const bool spacious = height_ >= 8;
  text_row_ = upper_ + (spacious ? 3 : 1);
  visible_ = std::max(0, height_ - (spacious ? 7 : 3));
  button_row_ = upper_ + height_ - (spacious ? 3 : 2);
  button_column_ = left_ + std::max(0, (width_ - 6) / 2);
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
  if (event.action == MouseAction::Scroll) {
    Scroll(-event.scroll_y);
  } else if (height_ >= 3 && width_ >= 8 && event.action == MouseAction::Press && event.button == MouseButton::Left &&
             event.row == button_row_ && event.column >= button_column_ && event.column < button_column_ + 6 &&
             event.column < left_ + width_ - 1) {
    acknowledged_ = true;
    Terminate();
  }
}

void MessageBox::OnUpdate() {
  if (!dirty_)
    return;
  dirty_ = false;
  for (int row = 0; row < Rows(); ++row)
    ClearRow(row);
  const int inside = std::max(0, width_ - 2);
  const std::string border = "+" + std::string(inside, '-') + "+";
  UpdateRow(upper_, left_, border);
  for (int row = 1; row < height_ - 1; ++row)
    UpdateRow(upper_ + row, left_, "|" + std::string(inside, ' ') + "|");
  if (height_ > 1)
    UpdateRow(upper_ + height_ - 1, left_, border);

  // Center the inverse title within the top border, measured in terminal cells.
  const auto title = detail::FitRow(" " + title_ + " ", inside).text;
  SetInverse(true);
  UpdateRow(upper_, left_ + (width_ - detail::TextWidth(title)) / 2, title);
  SetInverse(false);
  for (int row = 0; row < visible_ && top_ + row < int(lines_.size()); ++row)
    UpdateRow(text_row_ + row, text_column_, lines_[top_ + row]);
  if (height_ >= 3 && width_ >= 8) {
    SetInverse(true);
    UpdateRow(button_row_, button_column_, "[ OK ]");
    SetInverse(false);
  }
}
}  // namespace snowberg::tui
