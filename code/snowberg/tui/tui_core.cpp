#include "snowberg/tui/tui_core.h"

#include <algorithm>
#include <deque>
#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace snowberg::tui {
namespace {
class Terminal {
 public:
  bool Open(bool enhanced);
  ~Terminal();
  bool Read(std::deque<InputEvent> &events);

 private:
  bool active_{}, enhanced_{};
  detail::InputParser parser_;
#if defined(_WIN32)
  HANDLE input_{GetStdHandle(STD_INPUT_HANDLE)};
  HANDLE output_{GetStdHandle(STD_OUTPUT_HANDLE)};
  HANDLE screen_{INVALID_HANDLE_VALUE};
  DWORD input_mode_{}, output_mode_{};
  DWORD mouse_buttons_{};
  wchar_t high_surrogate_{};
  UINT input_code_page_{}, output_code_page_{};
  CONSOLE_CURSOR_INFO cursor_{};
#else
  termios original_{};
#endif
};

bool Terminal::Open(bool enhanced) {
  enhanced_ = enhanced;
#if defined(_WIN32)
  if (!GetConsoleMode(input_, &input_mode_) || !GetConsoleMode(output_, &output_mode_))
    return false;
  const DWORD mode =
      (input_mode_ | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT | ENABLE_EXTENDED_FLAGS | ENABLE_VIRTUAL_TERMINAL_INPUT) &
      ~(ENABLE_QUICK_EDIT_MODE | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
  if (!SetConsoleMode(input_, mode))
    return false;
  if (!SetConsoleMode(output_, output_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT)) {
    SetConsoleMode(input_, input_mode_);
    return false;
  }
  GetConsoleCursorInfo(output_, &cursor_);
  input_code_page_ = GetConsoleCP();
  output_code_page_ = GetConsoleOutputCP();
  if (!SetConsoleCP(CP_UTF8)) {
    SetConsoleMode(input_, input_mode_);
    SetConsoleMode(output_, output_mode_);
    return false;
  }
  if (!SetConsoleOutputCP(CP_UTF8)) {
    SetConsoleCP(input_code_page_);
    SetConsoleMode(input_, input_mode_);
    SetConsoleMode(output_, output_mode_);
    return false;
  }
#else
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
  // ConPTY forwards mouse tracking to the outer terminal only with VT input enabled.
  // Read() accepts both native console events and VT sequences from terminal bridges.
  std::cout << "\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1003h\x1b[?1006h";
  if (enhanced_)
    std::cout << "\x1b[>4;2m\x1b[>1u\x1b[?2004h";
  std::cout << std::flush;
#if defined(_WIN32)
  // Query dimensions and mouse coordinates from the active alternate screen.
  screen_ =
      CreateFileW(L"CONOUT$", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
#endif
  return static_cast<bool>(std::cout);
}

Terminal::~Terminal() {
  if (!active_)
    return;
  if (!std::cout)
    std::cout.clear();
  // Stop report production before draining: disabling SGR first could make
  // an in-flight mouse report fall back to the legacy encoding.
  std::cout << "\x1b[?1003l\x1b[?1000l\x1b[?1006l";
  if (enhanced_)
    std::cout << "\x1b[?2004l\x1b[<u\x1b[>4;0m";
  std::cout << std::flush;
#if defined(_WIN32)
  DWORD mode{};
  if (GetConsoleMode(input_, &mode))
    SetConsoleMode(input_, mode & ~ENABLE_MOUSE_INPUT);
#endif
  // Terminal bridges can still deliver reports already in transit. Discard
  // input until 40 ms of quiet, with a 250 ms cap under continuous traffic.
  // Never dispatch these events to application callbacks after Terminate().
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  while (std::chrono::steady_clock::now() < deadline) {
    const int timeout = std::min(40, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          deadline - std::chrono::steady_clock::now())
                                                          .count()));
    if (timeout <= 0)
      break;
#if defined(_WIN32)
    FlushConsoleInputBuffer(input_);
    if (WaitForSingleObject(input_, static_cast<DWORD>(timeout)) != WAIT_OBJECT_0)
      break;
#else
    pollfd descriptor{STDIN_FILENO, POLLIN, 0};
    const int ready = poll(&descriptor, 1, timeout);
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready <= 0 || (descriptor.revents & (POLLERR | POLLNVAL)))
      break;
    char discarded[4096];
    if (read(STDIN_FILENO, discarded, sizeof(discarded)) <= 0)
      break;
#endif
  }
  parser_ = {};
  std::cout << "\x1b[0m\x1b[?25h\x1b[?1049l" << std::flush;
#if defined(_WIN32)
  FlushConsoleInputBuffer(input_);
  if (screen_ != INVALID_HANDLE_VALUE)
    CloseHandle(screen_);
  SetConsoleMode(input_, input_mode_);
  SetConsoleMode(output_, output_mode_);
  SetConsoleCP(input_code_page_);
  SetConsoleOutputCP(output_code_page_);
  if (cursor_.dwSize)
    SetConsoleCursorInfo(output_, &cursor_);
#else
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_);
#endif
}

bool Terminal::Read(std::deque<InputEvent> &events) {
  parser_.Parse(events);
#if defined(_WIN32)
  const auto wait = WaitForSingleObject(input_, events.empty() ? 40 : 0);
  if (wait == WAIT_TIMEOUT) {
    parser_.Parse(events);
    return true;
  }
  if (wait != WAIT_OBJECT_0)
    return false;
  INPUT_RECORD records[128];
  DWORD count{};
  if (!ReadConsoleInputW(input_, records, 128, &count))
    return false;
  for (DWORD i = 0; i < count; ++i) {
    const auto &record = records[i];
    if (record.EventType == KEY_EVENT) {
      const auto &key = record.Event.KeyEvent;
      Modifiers modifiers{bool(key.dwControlKeyState & SHIFT_PRESSED),
                          bool(key.dwControlKeyState & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)),
                          bool(key.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))};
      // VT bridges deliver characters with no virtual key. Preserve sequence
      // order relative to native records by parsing after each feed.
      if (!key.wVirtualKeyCode && key.uChar.UnicodeChar > 0 && key.uChar.UnicodeChar < 128) {
        if (key.bKeyDown) {
          parser_.Feed(std::string(key.wRepeatCount, static_cast<char>(key.uChar.UnicodeChar)));
          parser_.Parse(events);
        }
        continue;
      }
      KeyEvent event{KeyFromVirtualKey(key.wVirtualKeyCode), 0, modifiers, bool(key.bKeyDown), key.wRepeatCount};
      char32_t code = key.uChar.UnicodeChar;
      if (code >= 0xd800 && code <= 0xdbff) {
        high_surrogate_ = static_cast<wchar_t>(code);
        continue;
      }
      if (code >= 0xdc00 && code <= 0xdfff) {
        code = high_surrogate_ ? 0x10000 + ((high_surrogate_ - 0xd800) << 10) + code - 0xdc00 : 0xfffd;
      }
      high_surrogate_ = 0;
      if (event.key == Key::Unknown && code) {
        event.key = Key::Character;
        event.character = code;
        if (code < 32) {
          event.modifiers.control = true;
          event.character = code == 0 ? U' ' : code <= 26 ? code + U'a' - 1 : code + 64;
        }
      }
      events.emplace_back(event);
    } else if (record.EventType == MOUSE_EVENT) {
      const auto &mouse = record.Event.MouseEvent;
      CONSOLE_SCREEN_BUFFER_INFO info{};
      if (!GetConsoleScreenBufferInfo(screen_ != INVALID_HANDLE_VALUE ? screen_ : output_, &info))
        continue;
      MouseEvent event;
      event.column = mouse.dwMousePosition.X - info.srWindow.Left;
      event.row = mouse.dwMousePosition.Y - info.srWindow.Top;
      event.modifiers = {bool(mouse.dwControlKeyState & SHIFT_PRESSED),
                         bool(mouse.dwControlKeyState & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)),
                         bool(mouse.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))};
      if (mouse.dwEventFlags == MOUSE_WHEELED || mouse.dwEventFlags == MOUSE_HWHEELED) {
        event.action = MouseAction::Scroll;
        const int delta = static_cast<SHORT>(HIWORD(mouse.dwButtonState));
        const int steps = delta / WHEEL_DELTA;
        if (mouse.dwEventFlags == MOUSE_WHEELED)
          event.scroll_y = steps;
        else
          event.scroll_x = steps;
        events.emplace_back(event);
      } else if (mouse.dwEventFlags == MOUSE_MOVED) {
        event.action = MouseAction::Move;
        event.button = mouse.dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED   ? MouseButton::Left
                       : mouse.dwButtonState & RIGHTMOST_BUTTON_PRESSED     ? MouseButton::Right
                       : mouse.dwButtonState & FROM_LEFT_2ND_BUTTON_PRESSED ? MouseButton::Middle
                                                                            : MouseButton::None;
        events.emplace_back(event);
      } else {
        const DWORD masks[] = {FROM_LEFT_1ST_BUTTON_PRESSED, FROM_LEFT_2ND_BUTTON_PRESSED, RIGHTMOST_BUTTON_PRESSED};
        const MouseButton buttons[] = {MouseButton::Left, MouseButton::Middle, MouseButton::Right};
        for (int j = 0; j < 3; ++j) {
          if (((mouse.dwButtonState ^ mouse_buttons_) & masks[j]) ||
              (mouse.dwEventFlags == DOUBLE_CLICK && (mouse.dwButtonState & masks[j]))) {
            event.button = buttons[j];
            event.action = mouse.dwButtonState & masks[j] ? MouseAction::Press : MouseAction::Release;
            events.emplace_back(event);
          }
        }
      }
      mouse_buttons_ = LOWORD(mouse.dwButtonState);
    }
  }
#else
  pollfd descriptor{STDIN_FILENO, POLLIN, 0};
  const int ready = poll(&descriptor, 1, events.empty() ? 40 : 0);
  if (ready < 0)
    return errno == EINTR;
  if (ready > 0) {
    if (descriptor.revents & (POLLERR | POLLNVAL))
      return false;
    char bytes[4096];
    const auto count = read(STDIN_FILENO, bytes, sizeof(bytes));
    if (count < 0)
      return errno == EINTR || errno == EAGAIN;
    if (count == 0)
      return false;
    parser_.Feed(std::string_view(bytes, static_cast<size_t>(count)));
  }
#endif
  parser_.Parse(events);
  return true;
}
}  // namespace

struct Core::Impl {
  bool active{}, running{}, failed{}, enhanced{};
  int columns{}, rows{};
  std::string foreground, background;
  bool inverse{};
};

Core::Core() : impl_(std::make_unique<Impl>()) {
}

Core::~Core() = default;

RunResult Core::Run() {
  if (impl_->active)
    return RunResult::Error;
  if (!IsTerminalAvailable())
    return RunResult::Unavailable;
  Terminal terminal;
  if (!terminal.Open(impl_->enhanced))
    return RunResult::Error;

  // Also reset Core state if a user callback unwinds; Terminal restores modes.
  struct ResetState {
    Impl &state;

    ~ResetState() {
      state.active = state.running = false;
    }
  } reset{*impl_};

  impl_->active = impl_->running = true;
  impl_->failed = false;
  impl_->foreground.clear();
  impl_->background.clear();
  impl_->inverse = false;
  auto refresh_size = [&] {
    int columns{}, rows{};
    if (!GetTerminalSize(columns, rows)) {
      impl_->failed = true;
      Terminate();
      return false;
    }
    columns = std::max(1, columns);
    rows = std::max(1, rows);
    const bool changed = columns != impl_->columns || rows != impl_->rows;
    impl_->columns = columns;
    impl_->rows = rows;
    return changed;
  };
  refresh_size();
  if (impl_->running)
    OnStart();
  if (impl_->running)
    OnResize(Columns(), Rows());
  if (impl_->running)
    OnUpdate();
  std::deque<InputEvent> events;
  while (impl_->running) {
    if (!terminal.Read(events)) {
      impl_->failed = true;
      break;
    }
    if (refresh_size() && impl_->running)
      OnResize(Columns(), Rows());
    while (impl_->running && !events.empty()) {
      const auto event = events.front();
      events.pop_front();
      if (const auto key = std::get_if<KeyEvent>(&event))
        OnKey(*key);
      else if (const auto mouse = std::get_if<MouseEvent>(&event))
        OnMouse(*mouse);
      else
        OnPaste(std::get<PasteEvent>(event));
    }
    if (impl_->running)
      OnUpdate();
  }
  OnStop();
  impl_->active = impl_->running = false;
  return impl_->failed ? RunResult::Error : RunResult::Completed;
}

void Core::EnableEnhancedInput(bool enabled) {
  if (!impl_->active)
    impl_->enhanced = enabled;
}

void Core::Terminate() {
  impl_->running = false;
}

int Core::Columns() const {
  return impl_->columns;
}

int Core::Rows() const {
  return impl_->rows;
}

void Core::SetForegroundColor(uint8_t r, uint8_t g, uint8_t b) {
  impl_->foreground = "\x1b[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

void Core::SetBackgroundColor(uint8_t r, uint8_t g, uint8_t b) {
  impl_->background = "\x1b[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

void Core::ResetColors() {
  impl_->foreground.clear();
  impl_->background.clear();
}

void Core::SetInverse(bool enabled) {
  impl_->inverse = enabled;
}

int Core::UpdateRow(int row, int start_column, std::string content) {
  if (!impl_->active || row < 0 || start_column < 0)
    return 0;
  // Query at the write boundary, not just once per tick: callbacks can run long.
  int columns{}, rows{};
  if (!GetTerminalSize(columns, rows)) {
    impl_->failed = true;
    Terminate();
    return -1;
  }
  if (row >= rows || start_column >= columns)
    return 0;
  const auto text = detail::FitRow(content, columns - start_column);
  if (!text.bytes)
    return 0;
  const auto prefix = "\x1b[" + std::to_string(row + 1) + ";" + std::to_string(start_column + 1) + "H\x1b[0m" +
                      impl_->foreground + impl_->background + (impl_->inverse ? "\x1b[7m" : "");
  // Explicitly reposition after the write to cancel deferred right-margin wrap.
  // No newline is emitted, including at the bottom-right cell.
  std::cout << prefix << text.text
            << "\x1b[0m\x1b[" + std::to_string(row + 1) + ";" + std::to_string(start_column + 1) + "H" << std::flush;
  if (!std::cout) {
    impl_->failed = true;
    Terminate();
    return -1;
  }
  return text.bytes;
}

void Core::ClearRow(int row) {
  if (!impl_->active || row < 0)
    return;
  int columns{}, rows{};
  if (!GetTerminalSize(columns, rows)) {
    impl_->failed = true;
    Terminate();
    return;
  }
  if (row >= rows)
    return;
  std::cout << "\x1b[" + std::to_string(row + 1) + ";1H\x1b[0m" << impl_->background << "\x1b[2K" << std::flush;
  if (!std::cout) {
    impl_->failed = true;
    Terminate();
  }
}
}  // namespace snowberg::tui
