#include "snowberg/tui/tui_util.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace snowberg::tui {
bool GetTerminalSize(int &columns, int &rows) {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO info{};
  // CONOUT$ refers to the current alternate buffer as well as the primary one.
  const auto screen =
      CreateFileW(L"CONOUT$", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
  const bool ok = screen != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(screen, &info);
  if (screen != INVALID_HANDLE_VALUE)
    CloseHandle(screen);
  if (!ok)
    return false;
  columns = info.srWindow.Right - info.srWindow.Left + 1;
  rows = info.srWindow.Bottom - info.srWindow.Top + 1;
#else
  winsize size{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) != 0)
    return false;
  columns = size.ws_col;
  rows = size.ws_row;
#endif
  return true;
}

bool IsTerminalAvailable() {
#if defined(_WIN32)
  DWORD input{}, output{};
  if (!GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &input) ||
      !GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &output))
    return false;
#else
  const auto term = std::getenv("TERM");
  if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO) || !term || !*term || std::string_view(term) == "dumb")
    return false;
#endif
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  for (;;) {
    int columns{}, rows{};
    if (!GetTerminalSize(columns, rows))
      return false;
    if (columns >= 2 && rows >= 2)
      return true;
#if defined(_WIN32)
    return false;
#else
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    poll(nullptr, 0, 10);
#endif
  }
}

Key MapVirtualKey(unsigned key) {
  switch (key) {
    case 0x08:
      return Key::Backspace;
    case 0x09:
      return Key::Tab;
    case 0x0d:
      return Key::Enter;
    case 0x1b:
      return Key::Escape;
    case 0x21:
      return Key::PageUp;
    case 0x22:
      return Key::PageDown;
    case 0x23:
      return Key::End;
    case 0x24:
      return Key::Home;
    case 0x25:
      return Key::Left;
    case 0x26:
      return Key::Up;
    case 0x27:
      return Key::Right;
    case 0x28:
      return Key::Down;
    case 0x2d:
      return Key::Insert;
    case 0x2e:
      return Key::Delete;
    default:
      if (key >= 0x70 && key <= 0x7b)
        return static_cast<Key>(static_cast<int>(Key::F1) + key - 0x70);
      return Key::Unknown;
  }
}

Key MapTerminalKey(char final, int parameter) {
  switch (final) {
    case 'A':
      return Key::Up;
    case 'B':
      return Key::Down;
    case 'C':
      return Key::Right;
    case 'D':
      return Key::Left;
    case 'H':
      return Key::Home;
    case 'F':
      return Key::End;
    case 'P':
      return Key::F1;
    case 'Q':
      return Key::F2;
    case 'R':
      return Key::F3;
    case 'S':
      return Key::F4;
    case 'Z':
      return Key::Tab;
    case '~':
      switch (parameter) {
        case 1:
        case 7:
          return Key::Home;
        case 2:
          return Key::Insert;
        case 3:
          return Key::Delete;
        case 4:
        case 8:
          return Key::End;
        case 5:
          return Key::PageUp;
        case 6:
          return Key::PageDown;
        case 11:
          return Key::F1;
        case 12:
          return Key::F2;
        case 13:
          return Key::F3;
        case 14:
          return Key::F4;
        case 15:
          return Key::F5;
        case 17:
          return Key::F6;
        case 18:
          return Key::F7;
        case 19:
          return Key::F8;
        case 20:
          return Key::F9;
        case 21:
          return Key::F10;
        case 23:
          return Key::F11;
        case 24:
          return Key::F12;
      }
  }
  return Key::Unknown;
}

namespace detail {
namespace {
struct Glyph {
  std::string_view bytes;
  int width;
  bool replace;
};

Glyph Next(std::string_view &text) {
  const auto first = static_cast<unsigned char>(text[0]);
  size_t length = first < 0x80 ? 1 : first >= 0xc2 && first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
  uint32_t code = first & (length == 1 ? 0x7f : length == 2 ? 0x1f : length == 3 ? 0x0f : 0x07);
  bool valid = first < 0x80 || (first >= 0xc2 && first <= 0xf4 && length <= text.size());
  for (size_t i = 1; valid && i < length; ++i) {
    const auto byte = static_cast<unsigned char>(text[i]);
    valid = (byte & 0xc0) == 0x80;
    code = (code << 6) | (byte & 0x3f);
  }
  if (valid && length > 1)
    valid = code >= (length == 2   ? 0x80u
                     : length == 3 ? 0x800u
                                   : 0x10000u) &&
            code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff);
  if (!valid) {
    text.remove_prefix(1);
    return {{}, 1, true};
  }
  const auto bytes = text.substr(0, length);
  text.remove_prefix(length);
  if (code < 0x20 || (code >= 0x7f && code < 0xa0) || code == 0x2028 || code == 0x2029)
    return {bytes, 1, true};
  if ((code >= 0x300 && code <= 0x36f) || (code >= 0xfe00 && code <= 0xfe0f))
    return {bytes, 0, false};
  const bool wide = (code >= 0x1100 && code <= 0x115f) || (code >= 0x2e80 && code <= 0xa4cf) ||
                    (code >= 0xac00 && code <= 0xd7a3) || (code >= 0xf900 && code <= 0xfaff) ||
                    (code >= 0xfe10 && code <= 0xfe6f) || (code >= 0xff01 && code <= 0xff60) ||
                    (code >= 0xffe0 && code <= 0xffe6) || (code >= 0x1f300 && code <= 0x1faff) ||
                    (code >= 0x20000 && code <= 0x3fffd);
  return {bytes, wide ? 2 : 1, false};
}
}  // namespace

RowText FitRow(std::string_view content, int cells) {
  RowText result;
  if (cells <= 0)
    return result;
  int used = 0;
  bool base = false;
  while (!content.empty()) {
    const auto before = content.size();
    const auto glyph = Next(content);
    const auto bytes = before - content.size();
    if (glyph.width > cells - used || bytes > static_cast<size_t>(std::numeric_limits<int>::max() - result.bytes))
      break;
    if (glyph.width || base)
      result.text += glyph.replace ? std::string_view(" ") : glyph.bytes;
    result.bytes += static_cast<int>(bytes);
    used += glyph.width;
    base = base || glyph.width > 0;
  }
  return result;
}

int TextWidth(std::string_view text) {
  int width = 0;
  while (!text.empty())
    width += Next(text).width;
  return width;
}

std::string TextSlice(std::string_view text, int offset, int width) {
  std::string result;
  int column = 0;
  bool include_combining = false;
  while (!text.empty()) {
    const auto glyph = Next(text);
    if (!glyph.width) {
      if (include_combining)
        result += glyph.bytes;
      continue;
    }
    include_combining = false;
    const int end = column + glyph.width;
    if (column >= offset + width)
      break;
    if (end > offset) {
      if (column >= offset && end <= offset + width) {
        result += glyph.replace ? std::string_view(" ") : glyph.bytes;
        include_combining = true;
      } else {
        result.append(std::max(0, std::min(end, offset + width) - std::max(column, offset)), ' ');
      }
    }
    column = end;
  }
  return result;
}

int ScrollOffset(int text_width, int viewport_width, double seconds) {
  const int distance = std::max(0, text_width - viewport_width);
  if (!distance || viewport_width <= 0)
    return 0;
  // Pause at both ends, then restart from the beginning without wrapping a row.
  const double travel = distance * 0.08;
  const double phase = std::fmod(std::max(0.0, seconds), 2.0 + travel);
  return std::clamp(static_cast<int>((phase - 1.0) / 0.08), 0, distance);
}

namespace {
bool Parameters(std::string_view text, std::vector<int> &values) {
  while (!text.empty()) {
    const auto end = text.find(';');
    const auto part = text.substr(0, end);
    int value{};
    const auto parsed = std::from_chars(part.data(), part.data() + part.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() || value < 0)
      return false;
    values.push_back(value);
    if (end == std::string_view::npos)
      break;
    text.remove_prefix(end + 1);
  }
  return true;
}
}  // namespace

void InputParser::Feed(std::string_view bytes, Clock::time_point now) {
  if (pending_.empty())
    pending_at_ = now;
  pending_.append(bytes);
}

void InputParser::Parse(std::deque<InputEvent> &events, Clock::time_point now) {
  while (!pending_.empty()) {
    const bool expired = now - pending_at_ >= std::chrono::milliseconds(80);
    const bool escape = pending_[0] == '\x1b';
    if (escape && pending_.size() == 1) {
      if (!expired)
        break;
      events.emplace_back(KeyEvent{Key::Escape});
      pending_.erase(0, 1);
      continue;
    }
    if (escape && (pending_[1] == '[' || pending_[1] == 'O')) {
      size_t end = 2;
      while (end < pending_.size() && (pending_[end] < '@' || pending_[end] > '~'))
        ++end;
      if (end == pending_.size()) {
        if (!expired && pending_.size() < 256)
          break;
        pending_.clear();  // Discard a truncated/oversized control sequence.
        continue;
      }
      const auto sequence = pending_.substr(2, end - 2);
      const char final = pending_[end];
      pending_.erase(0, end + 1);
      pending_at_ = now;
      std::vector<int> params;
      if (!sequence.empty() && sequence[0] == '<') {
        if ((final != 'M' && final != 'm') || !Parameters(std::string_view(sequence).substr(1), params) ||
            params.size() != 3 || params[1] < 1 || params[2] < 1)
          continue;
        const int button = params[0];
        MouseEvent event;
        event.column = params[1] - 1;
        event.row = params[2] - 1;
        event.modifiers = {bool(button & 4), bool(button & 8), bool(button & 16)};
        if (button & 64) {
          event.action = MouseAction::Scroll;
          if (button & 2)
            event.scroll_x = button & 1 ? 1 : -1;
          else
            event.scroll_y = button & 1 ? -1 : 1;
        } else {
          event.button = (button & 3) == 0   ? MouseButton::Left
                         : (button & 3) == 1 ? MouseButton::Middle
                         : (button & 3) == 2 ? MouseButton::Right
                                             : MouseButton::None;
          event.action = final == 'm' ? MouseAction::Release : (button & 32) ? MouseAction::Move : MouseAction::Press;
        }
        events.emplace_back(event);
      } else if (Parameters(sequence, params)) {
        const auto key = MapTerminalKey(final, params.empty() ? 0 : params[0]);
        if (key == Key::Unknown)
          continue;
        const int flags = params.size() > 1 && params[1] > 0 ? params[1] - 1 : 0;
        events.emplace_back(KeyEvent{key, 0, {bool(flags & 1) || final == 'Z', bool(flags & 2), bool(flags & 4)}});
      }
      continue;
    }
    const size_t start = escape ? 1 : 0;
    const auto first = static_cast<unsigned char>(pending_[start]);
    size_t length = first < 0x80                     ? 1
                    : first >= 0xc2 && first < 0xe0  ? 2
                    : first >= 0xe0 && first < 0xf0  ? 3
                    : first >= 0xf0 && first <= 0xf4 ? 4
                                                     : 1;
    if (pending_.size() < start + length && !expired)
      break;
    char32_t code = first & (length == 1 ? 0x7f : length == 2 ? 0x1f : length == 3 ? 0x0f : 7);
    bool valid = (length > 1 || first < 0x80) && pending_.size() >= start + length;
    for (size_t i = 1; valid && i < length; ++i) {
      const auto byte = static_cast<unsigned char>(pending_[start + i]);
      valid = (byte & 0xc0) == 0x80;
      code = (code << 6) | (byte & 0x3f);
    }
    valid = valid &&
            (length == 1 || code >= (length == 2   ? 0x80u
                                     : length == 3 ? 0x800u
                                                   : 0x10000u)) &&
            code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff);
    if (!valid) {
      code = 0xfffd;
      length = 1;
    }
    pending_.erase(0, start + length);
    pending_at_ = now;
    KeyEvent event{Key::Character, code, {false, escape, false}};
    if (code == '\r' || code == '\n')
      event.key = Key::Enter;
    else if (code == '\t')
      event.key = Key::Tab;
    else if (code == 8 || code == 127)
      event.key = Key::Backspace;
    else if (code < 32) {
      event.modifiers.control = true;
      event.character = code == 0 ? U' ' : code <= 26 ? code + U'a' - 1 : code + 64;
    }
    events.emplace_back(event);
  }
}
}  // namespace detail
}  // namespace snowberg::tui
