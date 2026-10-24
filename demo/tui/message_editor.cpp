#include "message_editor.h"

#include <algorithm>

#include "snowberg/tui/tui_util.h"

namespace tui_demo {
int CellWidth(char32_t character) {
  return snowberg::tui::detail::TextWidth(snowberg::tui::detail::EncodeUTF8(character));
}

std::string UTF8(std::u32string_view text) {
  std::string result;
  for (auto character : text)
    result += snowberg::tui::detail::EncodeUTF8(character);
  return result;
}

std::vector<TextLine> Wrap(std::u32string_view text, int width) {
  width = std::max(1, width);
  std::vector<TextLine> lines;
  size_t begin = 0;
  int column = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == U'\n') {
      lines.push_back({begin, i});
      begin = i + 1;
      column = 0;
      continue;
    }
    const int cells = std::min(width, CellWidth(text[i]));
    if (column && column + cells > width) {
      lines.push_back({begin, i});
      begin = i;
      column = 0;
    }
    column += cells;
  }
  lines.push_back({begin, text.size()});
  if (column == width)
    lines.push_back({text.size(), text.size()});
  return lines;
}

const std::u32string &Editor::Text() const {
  return text_;
}

size_t Editor::Cursor() const {
  return cursor_;
}

void Editor::Clear() {
  text_.clear();
  cursor_ = 0;
  preferred_column_.reset();
  upstream_ = false;
}

void Editor::Insert(std::u32string_view text) {
  std::u32string clean;
  for (size_t i = 0; i < text.size(); ++i) {
    auto c = text[i];
    if (c == U'\r') {
      if (i + 1 < text.size() && text[i + 1] == U'\n')
        ++i;
      c = U'\n';
    }
    if (c == U'\t')
      clean += U"    ";
    else if (c == U'\n' || (c >= 32 && !(c >= 127 && c < 160)))
      clean += c;
  }
  text_.insert(cursor_, clean);
  cursor_ += clean.size();
  preferred_column_.reset();
  upstream_ = false;
}

size_t Editor::Previous(size_t position) const {
  if (!position)
    return 0;
  --position;
  while (position && text_[position] != U'\n' && CellWidth(text_[position]) == 0)
    --position;
  return position;
}

size_t Editor::Next(size_t position) const {
  if (position >= text_.size())
    return text_.size();
  ++position;
  while (position < text_.size() && text_[position] != U'\n' && CellWidth(text_[position]) == 0)
    ++position;
  return position;
}

void Editor::Backspace() {
  const auto previous = Previous(cursor_);
  text_.erase(previous, cursor_ - previous);
  cursor_ = previous;
  preferred_column_.reset();
  upstream_ = false;
}

void Editor::Delete() {
  text_.erase(cursor_, Next(cursor_) - cursor_);
  preferred_column_.reset();
  upstream_ = false;
}

void Editor::MoveHorizontal(int direction) {
  cursor_ = direction < 0 ? Previous(cursor_) : Next(cursor_);
  preferred_column_.reset();
  upstream_ = false;
}

Position Editor::CursorPosition(int width) const {
  const auto lines = Wrap(text_, width);
  size_t row = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i].begin <= cursor_)
      row = i;
    else
      break;
  }
  if (upstream_ && row && lines[row - 1].end == cursor_ && lines[row].begin == cursor_)
    --row;
  int column = 0;
  for (size_t i = lines[row].begin; i < cursor_ && i < lines[row].end; ++i)
    column += std::min(std::max(1, width), CellWidth(text_[i]));
  return {static_cast<int>(row), column};
}

size_t Editor::Hit(const TextLine &line, int column, int width) const {
  int x = 0;
  for (size_t i = line.begin; i < line.end;) {
    const int cells = std::min(std::max(1, width), CellWidth(text_[i]));
    if (column < x + cells)
      return i;
    x += cells;
    i = Next(i);
  }
  return line.end;
}

void Editor::PlaceCursor(int row, int column, int width) {
  const auto lines = Wrap(text_, width);
  row = std::clamp(row, 0, static_cast<int>(lines.size()) - 1);
  cursor_ = Hit(lines[row], std::max(0, column), width);
  preferred_column_.reset();
  upstream_ = cursor_ == lines[row].end;
}

void Editor::MoveVertical(int direction, int width) {
  const auto position = CursorPosition(width);
  if (!preferred_column_)
    preferred_column_ = position.column;
  const auto lines = Wrap(text_, width);
  const int row = std::clamp(position.row + direction, 0, static_cast<int>(lines.size()) - 1);
  cursor_ = Hit(lines[row], *preferred_column_, width);
  upstream_ = cursor_ == lines[row].end;
}

void Editor::MoveLineEnd(bool end, int width) {
  const auto lines = Wrap(text_, width);
  const auto line = lines[CursorPosition(width).row];
  cursor_ = end ? line.end : line.begin;
  preferred_column_.reset();
  upstream_ = end;
}
}  // namespace tui_demo
