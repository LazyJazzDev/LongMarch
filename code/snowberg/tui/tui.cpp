#include "snowberg/tui/tui.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "snowberg/tui/tui_text.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace snowberg::tui {
namespace {
using Clock = std::chrono::steady_clock;

typedef enum { NONE, UP, DOWN, CONFIRM, CANCEL, CLICK } Action;

struct Event {
  Action action{NONE};
  int row{-1};
  unsigned repeat{1};
};

class Terminal {
 public:
  bool Open();
  ~Terminal();
  void Size(int &columns, int &rows) const;
  Event Read();

 private:
  bool active_{};
#if defined(_WIN32)
  HANDLE input_{GetStdHandle(STD_INPUT_HANDLE)};
  HANDLE output_{GetStdHandle(STD_OUTPUT_HANDLE)};
  DWORD input_mode_{}, output_mode_{};
  UINT code_page_{};
  CONSOLE_CURSOR_INFO cursor_{};
#else
  termios original_{};
  std::string pending_;
  Clock::time_point escape_time_{};
#endif
};

bool Terminal::Open() {
#if defined(_WIN32)
  if (!GetConsoleMode(input_, &input_mode_) || !GetConsoleMode(output_, &output_mode_))
    return false;
  const DWORD mode = (input_mode_ | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT | ENABLE_EXTENDED_FLAGS) &
                     ~(ENABLE_QUICK_EDIT_MODE | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT |
                       ENABLE_VIRTUAL_TERMINAL_INPUT);
  if (!SetConsoleMode(input_, mode))
    return false;
  if (!SetConsoleMode(output_, output_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT)) {
    SetConsoleMode(input_, input_mode_);
    return false;
  }
  GetConsoleCursorInfo(output_, &cursor_);
  code_page_ = GetConsoleOutputCP();
  SetConsoleOutputCP(CP_UTF8);
#else
  const char *term = std::getenv("TERM");
  if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO) || !term || std::string_view(term) == "dumb")
    return false;
  if (tcgetattr(STDIN_FILENO, &original_) != 0)
    return false;
  auto mode = original_;
  mode.c_lflag &= ~(ICANON | ECHO | ISIG | IEXTEN);
  mode.c_iflag &= ~(IXON | ICRNL);
  mode.c_cc[VMIN] = 0;
  mode.c_cc[VTIME] = 0;
  if (tcsetattr(STDIN_FILENO, TCSANOW, &mode) != 0)
    return false;
#endif
  active_ = true;
  std::cout << "\x1b[?1049h\x1b[?25l";
#if !defined(_WIN32)
  std::cout << "\x1b[?1000h\x1b[?1006h";
#endif
  std::cout << std::flush;
  return true;
}

Terminal::~Terminal() {
  if (!active_)
    return;
#if !defined(_WIN32)
  std::cout << "\x1b[?1006l\x1b[?1000l";
#endif
  std::cout << "\x1b[0m\x1b[?25h\x1b[?1049l" << std::flush;
#if defined(_WIN32)
  SetConsoleMode(input_, input_mode_);
  SetConsoleMode(output_, output_mode_);
  SetConsoleOutputCP(code_page_);
  if (cursor_.dwSize)
    SetConsoleCursorInfo(output_, &cursor_);
#else
  tcsetattr(STDIN_FILENO, TCSANOW, &original_);
#endif
}

void Terminal::Size(int &columns, int &rows) const {
  columns = 80;
  rows = 24;
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO info{};
  if (GetConsoleScreenBufferInfo(output_, &info)) {
    columns = info.srWindow.Right - info.srWindow.Left + 1;
    rows = info.srWindow.Bottom - info.srWindow.Top + 1;
  }
#else
  winsize size{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0) {
    columns = size.ws_col;
    rows = size.ws_row;
  }
#endif
  columns = std::max(1, columns);
  rows = std::max(1, rows);
}

Event Terminal::Read() {
#if defined(_WIN32)
  const auto wait = WaitForSingleObject(input_, 40);
  if (wait == WAIT_TIMEOUT)
    return {};
  if (wait != WAIT_OBJECT_0)
    return {CANCEL};
  INPUT_RECORD record{};
  DWORD count{};
  if (!ReadConsoleInputW(input_, &record, 1, &count))
    return {CANCEL};
  if (record.EventType == KEY_EVENT && record.Event.KeyEvent.bKeyDown) {
    const auto &key = record.Event.KeyEvent;
    if (key.wVirtualKeyCode == VK_UP)
      return {UP, -1, key.wRepeatCount};
    if (key.wVirtualKeyCode == VK_DOWN)
      return {DOWN, -1, key.wRepeatCount};
    if (key.wVirtualKeyCode == VK_RETURN)
      return {CONFIRM};
    if (key.wVirtualKeyCode == VK_ESCAPE || key.uChar.UnicodeChar == 3)
      return {CANCEL};
  } else if (record.EventType == MOUSE_EVENT) {
    const auto &mouse = record.Event.MouseEvent;
    if (mouse.dwEventFlags == MOUSE_WHEELED)
      return {static_cast<SHORT>(HIWORD(mouse.dwButtonState)) > 0 ? UP : DOWN};
    if ((mouse.dwEventFlags == 0 || mouse.dwEventFlags == DOUBLE_CLICK) &&
        (mouse.dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED)) {
      CONSOLE_SCREEN_BUFFER_INFO info{};
      if (GetConsoleScreenBufferInfo(output_, &info))
        return {CLICK, mouse.dwMousePosition.Y - info.srWindow.Top};
    }
  }
  return {};
#else
  pollfd descriptor{STDIN_FILENO, POLLIN, 0};
  const int ready = poll(&descriptor, 1, 40);
  if (ready < 0)
    return errno == EINTR ? Event{} : Event{CANCEL};
  if (ready > 0) {
    if (descriptor.revents & (POLLERR | POLLNVAL))
      return {CANCEL};
    char bytes[128];
    const auto count = read(STDIN_FILENO, bytes, sizeof(bytes));
    if (count <= 0)
      return {CANCEL};
    if (pending_.empty())
      escape_time_ = Clock::now();
    pending_.append(bytes, static_cast<size_t>(count));
  }
  if (pending_.empty())
    return {};
  const char first = pending_[0];
  if (first != '\x1b') {
    pending_.erase(0, 1);
    if (first == '\r' || first == '\n')
      return {CONFIRM};
    if (first == 3 || first == 4)
      return {CANCEL};
    return {};
  }
  if (pending_.size() >= 3 && (pending_[1] == '[' || pending_[1] == 'O')) {
    // CSI sequences can arrive across reads; consume one entire sequence at a time.
    size_t end = 2;
    while (end < pending_.size() && (pending_[end] < '@' || pending_[end] > '~'))
      ++end;
    if (end < pending_.size()) {
      const auto sequence = pending_.substr(0, end + 1);
      pending_.erase(0, end + 1);
      if (sequence.back() == 'A')
        return {UP};
      if (sequence.back() == 'B')
        return {DOWN};
      int button{}, x{}, y{};
      if (sequence.compare(0, 3, "\x1b[<") == 0 && sequence.back() == 'M' &&
          std::sscanf(sequence.c_str() + 3, "%d;%d;%d", &button, &x, &y) == 3) {
        if (button == 64)
          return {UP};
        if (button == 65)
          return {DOWN};
        if (button == 0)
          return {CLICK, y - 1};
      }
      return {};
    }
  }
  if (Clock::now() - escape_time_ >= std::chrono::milliseconds(80)) {
    pending_.clear();
    return {CANCEL};
  }
  return {};
#endif
}

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
}  // namespace

std::optional<size_t> Select(const std::string &title,
                             const std::vector<MenuOption> &options,
                             size_t initial_selection) {
  if (options.empty())
    return {};
  Terminal terminal;
  if (!terminal.Open())
    return SelectText(title, options);
  std::vector<std::string> labels;
  int name_width = 0;
  for (const auto &option : options)
    name_width = std::max(name_width, detail::TextWidth(option.name));
  for (const auto &option : options)
    labels.push_back(option.name + std::string(name_width - detail::TextWidth(option.name) + 2, ' ') +
                     option.description);
  size_t selected = std::min(initial_selection, options.size() - 1);
  size_t top = 0;
  auto selected_at = Clock::now();
  auto clicked_at = Clock::time_point{};
  size_t clicked = options.size();
  int old_columns = 0, old_rows = 0;
  std::string previous;
  for (;;) {
    int columns, rows;
    terminal.Size(columns, rows);
    if (columns != old_columns || rows != old_rows) {
      selected_at = Clock::now();
      old_columns = columns;
      old_rows = rows;
    }
    const int first_row = rows >= 4 ? 2 : 0;
    const int count = std::max(1, rows - first_row - (rows >= 4 ? 1 : 0));
    if (selected < top)
      top = selected;
    if (selected >= top + count)
      top = selected - count + 1;
    top = std::min(top, options.size() > static_cast<size_t>(count) ? options.size() - count : 0);
    // Leave the final column unused so a terminal never wraps an option into the next row.
    const int width = std::max(0, columns - 1);
    const int prefix = width >= 2 ? 2 : 0;
    std::string screen = "\x1b[H";
    if (first_row)
      screen += "\x1b[0m\x1b[2K" + detail::TextSlice(title, 0, width) + "\x1b[2;1H\x1b[2K";
    for (int row = 0; row < count; ++row) {
      screen += "\x1b[" + std::to_string(first_row + row + 1) + ";1H\x1b[0m\x1b[2K";
      const size_t index = top + row;
      if (index >= options.size())
        continue;
      const bool active = index == selected;
      if (active)
        screen += "\x1b[7m";
      if (prefix)
        screen += active ? "> " : "  ";
      const double seconds = std::chrono::duration<double>(Clock::now() - selected_at).count();
      const int offset = active ? detail::ScrollOffset(detail::TextWidth(labels[index]), width - prefix, seconds) : 0;
      const auto text = detail::TextSlice(labels[index], offset, width - prefix);
      screen += text + std::string(std::max(0, width - prefix - detail::TextWidth(text)), ' ') + "\x1b[0m";
    }
    if (first_row) {
      const auto hint = "Up/Down: select  Enter: confirm  Esc: exit  Click: select  Double-click: confirm  [" +
                        std::to_string(selected + 1) + "/" + std::to_string(options.size()) + "]";
      screen += "\x1b[" + std::to_string(rows) + ";1H\x1b[0m\x1b[2K" + detail::TextSlice(hint, 0, width);
    }
    if (screen != previous) {
      std::cout << screen << std::flush;
      if (!std::cout)
        return {};
      previous = std::move(screen);
    }
    const auto event = terminal.Read();
    const auto before = selected;
    if (event.action == CANCEL)
      return {};
    if (event.action == CONFIRM)
      return selected;
    if (event.action == UP && selected > 0)
      selected -= std::min(selected, static_cast<size_t>(event.repeat));
    if (event.action == DOWN && selected + 1 < options.size())
      selected += std::min(options.size() - selected - 1, static_cast<size_t>(event.repeat));
    if (event.action == CLICK && event.row >= first_row && event.row < first_row + count) {
      const auto index = top + event.row - first_row;
      if (index < options.size()) {
        const auto now = Clock::now();
        if (clicked == index && selected == index && now - clicked_at < std::chrono::milliseconds(400))
          return index;
        selected = index;
        clicked = index;
        clicked_at = now;
      }
    } else if (event.action != NONE) {
      clicked = options.size();
    }
    if (before != selected)
      selected_at = Clock::now();
  }
}
}  // namespace snowberg::tui
