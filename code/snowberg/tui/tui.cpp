#include "snowberg/tui/tui.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>

#include "snowberg/tui/tui_text.h"

namespace snowberg::tui {
namespace {
using Clock = std::chrono::steady_clock;

std::optional<size_t> SelectText(const std::string &title, const std::vector<MenuOption> &options) {
  std::cout << '\n' << title << '\n';
  for (size_t i = 0; i < options.size(); ++i)
    std::cout << "  " << i + 1 << ". " << options[i].name << "  " << options[i].description << '\n';
  for (;;) {
    std::cout << "\nSelect option [1-" << options.size() << " or name], q to quit: " << std::flush;
    std::string input;
    if (!std::getline(std::cin, input))
      return {};
    const auto begin = input.find_first_not_of(" \t\r");
    if (begin == std::string::npos)
      continue;
    input = input.substr(begin, input.find_last_not_of(" \t\r") - begin + 1);
    if (input == "q" || input == "quit" || input == "\x1b")
      return {};
    for (size_t i = 0; i < options.size(); ++i)
      if (input == options[i].name)
        return i;
    size_t number{};
    const auto result = std::from_chars(input.data(), input.data() + input.size(), number);
    if (result.ec == std::errc{} && result.ptr == input.data() + input.size() && number > 0 && number <= options.size())
      return number - 1;
    std::cout << "Invalid selection. Enter a listed number or name.\n";
  }
}

class Menu : public Core {
 public:
  Menu(const std::string &title, const std::vector<MenuOption> &options, size_t initial)
      : title_(title),
        selected_(std::min(initial, options.size() - 1)) {
    int name_width = 0;
    for (const auto &option : options)
      name_width = std::max(name_width, detail::TextWidth(option.name));
    for (const auto &option : options)
      labels_.push_back(option.name + std::string(name_width - detail::TextWidth(option.name) + 2, ' ') +
                        option.description);
  }

  std::optional<size_t> Result() const {
    return result_;
  }

 protected:
  void OnResize(int, int) override {
    previous_.clear();
    selected_at_ = Clock::now();
    Layout();
  }

  void OnKey(const KeyEvent &event) override {
    if (!event.pressed)
      return;
    clicked_ = labels_.size();
    if (event.key == Key::Escape || (event.key == Key::Character && event.modifiers.control &&
                                     (event.character == U'c' || event.character == U'd'))) {
      Terminate();
    } else if (event.key == Key::Enter) {
      result_ = selected_;
      Terminate();
    } else if (event.key == Key::Up) {
      SelectIndex(selected_ - std::min(selected_, static_cast<size_t>(event.repeat)));
    } else if (event.key == Key::Down) {
      SelectIndex(selected_ + std::min(labels_.size() - selected_ - 1, static_cast<size_t>(event.repeat)));
    }
  }

  void OnMouse(const MouseEvent &event) override {
    if (event.action == MouseAction::Scroll) {
      clicked_ = labels_.size();
      if (event.scroll_y > 0)
        SelectIndex(selected_ - std::min(selected_, static_cast<size_t>(event.scroll_y)));
      else if (event.scroll_y < 0)
        SelectIndex(selected_ + std::min(labels_.size() - selected_ - 1, static_cast<size_t>(-event.scroll_y)));
    } else if (event.action == MouseAction::Press && event.button == MouseButton::Left && event.column >= 0 &&
               event.column < Columns() && event.row >= first_row_ && event.row < first_row_ + count_) {
      const size_t index = top_ + event.row - first_row_;
      if (index < labels_.size()) {
        const auto now = Clock::now();
        if (clicked_ == index && selected_ == index && now - clicked_at_ < std::chrono::milliseconds(400)) {
          result_ = index;
          Terminate();
          return;
        }
        SelectIndex(index);
        clicked_ = index;
        clicked_at_ = now;
      }
    }
  }

  void OnUpdate() override {
    Layout();
    const int width = std::max(0, Columns() - 1);
    const int prefix = width >= 2 ? 2 : 0;
    std::vector<std::pair<std::string, bool>> lines(Rows());
    if (first_row_)
      lines[0].first = detail::TextSlice(title_, 0, width);
    for (int row = 0; row < count_; ++row) {
      const size_t index = top_ + row;
      if (index >= labels_.size())
        continue;
      auto &line = lines[first_row_ + row];
      line.second = index == selected_;
      if (prefix)
        line.first = line.second ? "> " : "  ";
      const double seconds = std::chrono::duration<double>(Clock::now() - selected_at_).count();
      const int offset =
          line.second ? detail::ScrollOffset(detail::TextWidth(labels_[index]), width - prefix, seconds) : 0;
      line.first += detail::TextSlice(labels_[index], offset, width - prefix);
    }
    if (first_row_) {
      lines.back().first =
          detail::TextSlice("Up/Down: select  Enter: confirm  Esc: exit  Click: select  Double-click: confirm  [" +
                                std::to_string(selected_ + 1) + "/" + std::to_string(labels_.size()) + "]",
                            0, width);
    }
    for (int row = 0; row < Rows(); ++row) {
      auto &line = lines[row];
      line.first.append(std::max(0, width - detail::TextWidth(line.first)), ' ');
      if (row >= static_cast<int>(previous_.size()) || previous_[row] != line) {
        SetInverse(line.second);
        UpdateRow(row, 0, line.first);
      }
    }
    previous_ = std::move(lines);
  }

 private:
  void Layout() {
    first_row_ = Rows() >= 4 ? 2 : 0;
    count_ = std::max(1, Rows() - first_row_ - (Rows() >= 4 ? 1 : 0));
    if (selected_ < top_)
      top_ = selected_;
    if (selected_ >= top_ + count_)
      top_ = selected_ - count_ + 1;
    top_ = std::min(top_, labels_.size() > static_cast<size_t>(count_) ? labels_.size() - count_ : 0);
  }

  void SelectIndex(size_t index) {
    if (selected_ != index)
      selected_at_ = Clock::now();
    selected_ = index;
    Layout();
  }

  std::string title_;
  std::vector<std::string> labels_;
  size_t selected_{}, top_{}, clicked_{static_cast<size_t>(-1)};
  int first_row_{}, count_{};
  Clock::time_point selected_at_{Clock::now()}, clicked_at_{};
  std::vector<std::pair<std::string, bool>> previous_;
  std::optional<size_t> result_;
};
}  // namespace

std::optional<size_t> Select(const std::string &title,
                             const std::vector<MenuOption> &options,
                             size_t initial_selection) {
  if (options.empty())
    return {};
  Menu menu(title, options, initial_selection);
  const auto status = menu.Run();
  if (status == RunResult::Unavailable)
    return SelectText(title, options);
  return status == RunResult::Completed ? menu.Result() : std::nullopt;
}
}  // namespace snowberg::tui
