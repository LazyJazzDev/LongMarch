#include <algorithm>
#include <chrono>
#include <iostream>
#include <string_view>

#include "message_editor.h"
#include "snowberg/tui/tui.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
using namespace snowberg::tui;
using Clock = std::chrono::steady_clock;

class Messages : public Core {
 public:
  Messages() {
    EnableEnhancedInput();
  }

  bool Dump() const {
#if defined(_WIN32)
    // Core has restored the caller's code page. Dump remains UTF-8, including
    // when redirected, but a native console needs UTF-8 enabled for this write.
    DWORD mode{};
    const auto console = GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode);
    const auto code_page = GetConsoleOutputCP();
    if (console) {
      std::cout << std::flush;
      if (!SetConsoleOutputCP(CP_UTF8))
        return false;
    }
#endif
    for (size_t i = 0; i < messages_.size(); ++i)
      std::cout << "--- Message " << i + 1 << " ---\n" << tui_demo::UTF8(messages_[i]) << '\n';
#if defined(_WIN32)
    if (console) {
      std::cout << std::flush;
      if (!SetConsoleOutputCP(code_page))
        return false;
    }
#endif
    return static_cast<bool>(std::cout);
  }

 protected:
  void OnResize(int, int) override {
    previous_.clear();
    ReflowHistory();
    Changed();
  }

  void OnKey(const KeyEvent &event) override {
    if (!event.pressed)
      return;
    if (event.key == Key::Escape ||
        (event.key == Key::Character && event.modifiers.control && event.character == U'c')) {
      Terminate();
      return;
    }
    for (unsigned i = 0; i < event.repeat; ++i) {
      switch (event.key) {
        case Key::Enter:
          if (event.modifiers.shift || event.modifiers.alt || event.character == U'\n')
            editor_.Insert(U"\n");
          else
            Send();
          break;
        case Key::Character:
          if (event.modifiers.control && event.character == U'j')
            editor_.Insert(U"\n");
          else if (!event.modifiers.control && !event.modifiers.alt)
            editor_.Insert(std::u32string(1, event.character));
          break;
        case Key::Backspace:
          editor_.Backspace();
          break;
        case Key::Delete:
          editor_.Delete();
          break;
        case Key::Left:
          editor_.MoveHorizontal(-1);
          break;
        case Key::Right:
          editor_.MoveHorizontal(1);
          break;
        case Key::Up:
          editor_.MoveVertical(-1, TextWidth());
          break;
        case Key::Down:
          editor_.MoveVertical(1, TextWidth());
          break;
        case Key::Home:
          editor_.MoveLineEnd(false, TextWidth());
          break;
        case Key::End:
          editor_.MoveLineEnd(true, TextWidth());
          break;
        case Key::PageUp:
          Scroll(-std::max(1, HistoryHeight() - 1));
          break;
        case Key::PageDown:
          Scroll(std::max(1, HistoryHeight() - 1));
          break;
        case Key::Tab:
          editor_.Insert(U"    ");
          break;
        default:
          break;
      }
    }
    Changed();
  }

  void OnPaste(const PasteEvent &event) override {
    editor_.Insert(detail::DecodeUTF8(event.text));
    Changed();
  }

  void OnMouse(const MouseEvent &event) override {
    if (event.action == MouseAction::Scroll) {
      Scroll(-event.scroll_y * 3);
      dirty_ = true;
    } else if (event.action == MouseAction::Press && event.button == MouseButton::Left) {
      Layout();
      if (event.row >= input_top_ + 1 && event.row < Rows() - 2 && event.column >= 0 && event.column < Columns()) {
        editor_.PlaceCursor(input_scroll_ + event.row - input_top_ - 1, event.column - 2, TextWidth());
        Changed();
      }
    }
  }

  void OnUpdate() override {
    const bool blink =
        (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - edited_at_).count() / 500) % 2 == 0;
    if (!dirty_ && blink == last_blink_)
      return;
    Layout();
    const std::string hint = " Enter send | Shift/Alt+Enter newline | PgUp/PgDn history | Esc exit";
    std::vector<std::string> frame(Rows());
    const bool usable = Columns() >= 12 && Rows() >= 6;
    if (!usable) {
      frame[0] = "Resize to at least 12 columns and 6 rows";
    } else {
      frame[0] = " TUI Messages | " + std::to_string(messages_.size()) + " recorded | " +
                 (follow_tail_ ? "Latest" : "History");
      if (history_.empty())
        frame[1] = " No messages yet. Type below and press Enter.";
      for (int row = 0; row < HistoryHeight(); ++row) {
        const int index = history_top_ + row;
        if (index >= 0 && index < static_cast<int>(history_.size()))
          frame[row + 1] = history_[index];
      }
      const auto lines = tui_demo::Wrap(editor_.Text(), TextWidth());
      for (int row = 0; row < edit_rows_; ++row) {
        const size_t index = input_scroll_ + row;
        if (index >= lines.size())
          break;
        const auto line = lines[index];
        frame[input_top_ + 1 + row] =
            (row == 0 ? "> " : "  ") +
            tui_demo::UTF8(std::u32string_view(editor_.Text()).substr(line.begin, line.end - line.begin));
      }
      frame.back() = hint;
    }
    const auto caret = editor_.CursorPosition(TextWidth());
    const int caret_row = usable ? input_top_ + 1 + caret.row - input_scroll_ : -1;
    for (int row = 0; row < Rows(); ++row) {
      const bool input = usable && row >= input_top_ && row < Rows() - 1;
      auto text = detail::TextSlice(frame[row], 0, Columns());
      text.append(std::max(0, Columns() - detail::TextWidth(text)), ' ');
      frame[row] = std::move(text);
      // Restore the old cursor cell with its full row before painting the new one.
      if (row >= static_cast<int>(previous_.size()) || frame[row] != previous_[row] || row == caret_row ||
          row == last_caret_row_ || input_top_ != previous_input_top_) {
        SetForegroundColor(input ? 235 : 210, input ? 239 : 217, input ? 245 : 226);
        SetBackgroundColor(input ? 43 : 20, input ? 50 : 24, input ? 65 : 32);
        if (usable && row == Rows() - 1)
          SetForegroundColor(160, 170, 185);
        UpdateRow(row, 0, frame[row]);
        if (usable && row == Rows() - 1) {
          SetForegroundColor(236, 214, 150);
          for (const auto key : {"Enter", "Shift/Alt+Enter", "PgUp/PgDn", "Esc"}) {
            const auto column = hint.find(key);
            if (column < static_cast<size_t>(Columns()))
              UpdateRow(row, static_cast<int>(column), key);
          }
        }
      }
    }
    if (usable && blink && caret_row >= input_top_ + 1 && caret_row < Rows() - 2) {
      std::string glyph = " ";
      if (caret.column < TextWidth() && editor_.Cursor() < editor_.Text().size() &&
          editor_.Text()[editor_.Cursor()] != U'\n') {
        size_t end = editor_.Cursor() + 1;
        while (end < editor_.Text().size() && tui_demo::CellWidth(editor_.Text()[end]) == 0)
          ++end;
        glyph = tui_demo::UTF8(std::u32string_view(editor_.Text()).substr(editor_.Cursor(), end - editor_.Cursor()));
      }
      SetForegroundColor(25, 31, 41);
      SetBackgroundColor(226, 232, 240);
      UpdateRow(caret_row, 2 + caret.column, glyph);
    }
    previous_ = std::move(frame);
    previous_input_top_ = input_top_;
    last_caret_row_ = caret_row;
    last_blink_ = blink;
    dirty_ = false;
  }

 private:
  int TextWidth() const {
    return std::max(1, Columns() - 4);
  }

  int HistoryHeight() const {
    return std::max(0, input_top_ - 1);
  }

  void Changed() {
    dirty_ = true;
    edited_at_ = Clock::now();
  }

  void Layout() {
    const auto lines = tui_demo::Wrap(editor_.Text(), TextWidth());
    const int maximum = std::max(1, std::min(8, Rows() / 2 - 2));
    edit_rows_ = std::clamp(static_cast<int>(lines.size()), 1, maximum);
    input_top_ = std::max(0, Rows() - edit_rows_ - 3);
    const auto caret = editor_.CursorPosition(TextWidth());
    input_scroll_ = std::clamp(input_scroll_, std::max(0, caret.row - edit_rows_ + 1), caret.row);
    const int last = std::max(0, static_cast<int>(history_.size()) - HistoryHeight());
    history_top_ = follow_tail_ ? last : std::clamp(history_top_, 0, last);
  }

  void ReflowHistory() {
    history_.clear();
    for (size_t i = 0; i < messages_.size(); ++i) {
      history_.push_back(" [" + std::to_string(i + 1) + "]");
      for (auto line : tui_demo::Wrap(messages_[i], TextWidth()))
        history_.push_back("  " +
                           tui_demo::UTF8(std::u32string_view(messages_[i]).substr(line.begin, line.end - line.begin)));
      history_.emplace_back();
    }
  }

  void Scroll(int delta) {
    Layout();
    const int last = std::max(0, static_cast<int>(history_.size()) - HistoryHeight());
    history_top_ = std::clamp(history_top_ + delta, 0, last);
    follow_tail_ = history_top_ == last;
  }

  void Send() {
    if (editor_.Text().empty())
      return;
    messages_.push_back(editor_.Text());
    editor_.Clear();
    input_scroll_ = 0;
    follow_tail_ = true;
    ReflowHistory();
  }

  tui_demo::Editor editor_;
  std::vector<std::u32string> messages_;
  std::vector<std::string> history_, previous_;
  int input_top_{}, edit_rows_{1}, input_scroll_{}, history_top_{};
  int previous_input_top_{-1}, last_caret_row_{-1};
  bool follow_tail_{true}, dirty_{true}, last_blink_{};
  Clock::time_point edited_at_{Clock::now()};
};
}  // namespace

int main(int argc, char **argv) {
  bool dump = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--dump")
      dump = true;
    else if (std::string_view(argv[i]) == "--help") {
      std::cout << "TUI message recorder\nEnter sends; Shift+Enter / Alt+Enter / Ctrl+J inserts a newline.\n"
                   "Arrows and mouse click move the cursor; Backspace/Delete edit.\n"
                   "Mouse wheel or PageUp/PageDown scroll history; Esc/Ctrl+C exits.\n"
                   "--dump prints recorded messages after exit. Messages are session-local.\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << '\n';
      return 1;
    }
  }
  Messages app;
  const auto result = app.Run();
  if (result == RunResult::Unavailable) {
    std::cerr
        << "This demo needs an interactive terminal. In CLion Run, enable Emulate terminal in the output console.\n";
    return 1;
  }
  if (result == RunResult::Error) {
    std::cerr << "Terminal input or output failed.\n";
    return 1;
  }
  if (dump) {
    if (!app.Dump()) {
      std::cerr << "Message output failed.\n";
      return 1;
    }
  }
  return 0;
}
