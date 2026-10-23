#pragma once

#include <optional>
#include <string>
#include <vector>

namespace tui_demo {
struct TextLine {
  size_t begin{}, end{};
};

struct Position {
  int row{}, column{};
};

int CellWidth(char32_t character);
std::string UTF8(std::u32string_view text);
std::vector<TextLine> Wrap(std::u32string_view text, int width);

class Editor {
 public:
  const std::u32string &Text() const;
  size_t Cursor() const;
  void Insert(std::u32string_view text);
  void Clear();
  void Backspace();
  void Delete();
  void MoveHorizontal(int direction);
  void MoveVertical(int direction, int width);
  void MoveLineEnd(bool end, int width);
  void PlaceCursor(int row, int column, int width);
  Position CursorPosition(int width) const;

 private:
  size_t Previous(size_t position) const;
  size_t Next(size_t position) const;
  size_t Hit(const TextLine &line, int column, int width) const;
  std::u32string text_;
  size_t cursor_{};
  bool upstream_{};
  std::optional<int> preferred_column_;
};
}  // namespace tui_demo
