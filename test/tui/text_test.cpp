#include <gtest/gtest.h>

#include "snowberg/tui/tui_text.h"

namespace snowberg::tui::detail {
TEST(TuiText, CropsToCellsWithoutSplittingWideCharacters) {
  EXPECT_EQ(TextWidth(u8"A中文B"), 6);
  EXPECT_EQ(TextSlice(u8"A中文B", 0, 2), "A ");
  EXPECT_EQ(TextSlice(u8"A中文B", 2, 3), u8" 文");
  EXPECT_EQ(TextSlice(u8"A中文B", 5, 1), "B");
  EXPECT_EQ(TextSlice(u8"A中文B", 0, 0), "");
}

TEST(TuiText, KeepsCombiningMarksWithTheirVisibleBase) {
  EXPECT_EQ(TextWidth(u8"e\u0301x"), 2);
  EXPECT_EQ(TextSlice(u8"e\u0301x", 0, 1), u8"e\u0301");
  EXPECT_EQ(TextSlice(u8"e\u0301x", 1, 1), "x");
}

TEST(TuiText, PreventsControlCharactersFromWritingOtherRows) {
  EXPECT_EQ(TextSlice("a\nb\rc\td\x1b[2J", 0, 30), "a b c d [2J");
  EXPECT_EQ(TextSlice("x\xff\xe4", 0, 20), "x  ");
}

TEST(TuiText, SelectedTextPausesAtBothEndsThenRestarts) {
  EXPECT_EQ(ScrollOffset(20, 10, 0.5), 0);
  EXPECT_EQ(ScrollOffset(20, 10, 1.5), 6);
  EXPECT_EQ(ScrollOffset(20, 10, 2.5), 10);
  EXPECT_EQ(ScrollOffset(20, 10, 2.9), 0);
  EXPECT_EQ(ScrollOffset(5, 10, 20), 0);
  EXPECT_EQ(ScrollOffset(20, 0, 20), 0);
}
}  // namespace snowberg::tui::detail
