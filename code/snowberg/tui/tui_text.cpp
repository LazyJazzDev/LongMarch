#include "snowberg/tui/tui_text.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "snowberg/tui/tui_util.h"

namespace snowberg::tui::detail {
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
}  // namespace snowberg::tui::detail
