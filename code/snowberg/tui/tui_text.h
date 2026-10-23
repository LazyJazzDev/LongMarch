#pragma once
#include <string>
#include <string_view>

namespace snowberg::tui::detail {
// Terminal-cell coordinates, not UTF-8 byte offsets. Control characters are sanitized.
int TextWidth(std::string_view text);
std::string TextSlice(std::string_view text, int offset, int width);
int ScrollOffset(int text_width, int viewport_width, double seconds);
}  // namespace snowberg::tui::detail
