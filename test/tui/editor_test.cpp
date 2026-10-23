#include <gtest/gtest.h>

#include "message_editor.h"

namespace tui_demo {
TEST(TuiEditor, UnicodeInsertionDeletionAndLineJoining) {
  Editor editor;
  editor.Insert(U"A中e\u0301\nB");
  editor.MoveHorizontal(-1);
  editor.Backspace();
  EXPECT_EQ(editor.Text(), U"A中e\u0301B");
  editor.Backspace();
  EXPECT_EQ(editor.Text(), U"A中B");
  editor.MoveHorizontal(-1);
  editor.Delete();
  EXPECT_EQ(editor.Text(), U"AB");
  editor.Insert(U"文");
  EXPECT_EQ(editor.Text(), U"A文B");
}

TEST(TuiEditor, WrappedVerticalMovementKeepsPreferredCell) {
  Editor editor;
  editor.Insert(U"abcdef\nx\nabcdef");
  editor.MoveVertical(-1, 10);
  EXPECT_EQ(editor.CursorPosition(10).column, 1);
  editor.MoveVertical(-1, 10);
  EXPECT_EQ(editor.CursorPosition(10).column, 6);
  editor.MoveVertical(1, 10);
  EXPECT_EQ(editor.CursorPosition(10).row, 1);
  editor.Clear();
  editor.Insert(U"abcdefgh");
  editor.MoveVertical(-1, 4);
  EXPECT_EQ(editor.CursorPosition(4).row, 1);
  editor.MoveLineEnd(true, 4);
  EXPECT_EQ(editor.CursorPosition(4).row, 1);
  EXPECT_EQ(editor.CursorPosition(4).column, 4);
  editor.MoveVertical(-1, 4);
  EXPECT_EQ(editor.CursorPosition(4).row, 0);
  EXPECT_EQ(editor.CursorPosition(4).column, 4);
}

TEST(TuiEditor, MouseHitsCellsNotUnicodeBytes) {
  Editor editor;
  editor.Insert(U"A中文B");
  editor.PlaceCursor(0, 2, 20);
  EXPECT_EQ(editor.Cursor(), 1);
  editor.Insert(U"X");
  EXPECT_EQ(editor.Text(), U"AX中文B");
  editor.PlaceCursor(50, 100, 4);
  EXPECT_EQ(editor.Cursor(), editor.Text().size());
}

TEST(TuiEditor, PasteNormalizesNewlinesAndControls) {
  Editor editor;
  editor.Insert(U"a\r\nb\rc\t\x1b");
  EXPECT_EQ(editor.Text(), U"a\nb\nc    ");
  editor.Clear();
  EXPECT_TRUE(editor.Text().empty());
  EXPECT_EQ(editor.Cursor(), 0);
}
}  // namespace tui_demo
