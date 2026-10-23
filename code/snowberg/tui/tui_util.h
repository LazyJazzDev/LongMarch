#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <variant>

namespace snowberg::tui {

enum class Key {
  Unknown,
  Character,
  Escape,
  Enter,
  Tab,
  Backspace,
  Up,
  Down,
  Left,
  Right,
  Home,
  End,
  Insert,
  Delete,
  PageUp,
  PageDown,
  F1,
  F2,
  F3,
  F4,
  F5,
  F6,
  F7,
  F8,
  F9,
  F10,
  F11,
  F12
};

struct Modifiers {
  bool shift{}, alt{}, control{};
};

struct KeyEvent {
  Key key{Key::Unknown};
  char32_t character{};
  Modifiers modifiers{};
  bool pressed{true};
  unsigned repeat{1};
};
enum class MouseAction { Move, Press, Release, Scroll };
enum class MouseButton { None, Left, Middle, Right };

struct MouseEvent {
  MouseAction action{MouseAction::Move};
  MouseButton button{MouseButton::None};
  int column{}, row{};         // Zero-based terminal cells, relative to the viewport.
  int scroll_x{}, scroll_y{};  // Positive means right/up; measured in wheel steps.
  Modifiers modifiers{};
};

struct PasteEvent {
  std::string text;
};

using InputEvent = std::variant<KeyEvent, MouseEvent, PasteEvent>;

// Read-only capability check. POSIX waits up to 500 ms for an IDE PTY's size.
bool IsTerminalAvailable();
bool GetTerminalSize(int &columns, int &rows);
Key KeyFromVirtualKey(unsigned key);  // Windows virtual-key values; no platform headers required.
Key MapTerminalKey(char final, int parameter = 0);

namespace detail {
std::string EncodeUTF8(char32_t character);
std::u32string DecodeUTF8(std::string_view text);
// Terminal-cell coordinates, not UTF-8 byte offsets. Control characters are sanitized.
int TextWidth(std::string_view text);
std::string TextSlice(std::string_view text, int offset, int width);
int ScrollOffset(int text_width, int viewport_width, double seconds);

// Clips a UTF-8 prefix to cells, sanitizes controls/invalid bytes, and reports
// consumed source bytes. A wide glyph that cannot fit is left unconsumed.
struct RowText {
  std::string text;
  int bytes{};
};

RowText FitRow(std::string_view content, int cells);

// Incremental VT decoder. Emits all complete events in arrival order.
class InputParser {
 public:
  using Clock = std::chrono::steady_clock;
  void Feed(std::string_view bytes, Clock::time_point now = Clock::now());
  void Parse(std::deque<InputEvent> &events, Clock::time_point now = Clock::now());

 private:
  std::string pending_;
  bool pasting_{};
  Clock::time_point pending_at_{};
};
}  // namespace detail
}  // namespace snowberg::tui
