#include <gtest/gtest.h>

#include "snowberg/tui/tui_util.h"

namespace snowberg::tui {
TEST(TuiUtil, RowPrefixReportsSourceBytesAndNeverSplitsGlyphs) {
  auto text = detail::FitRow(u8"A中文B", 4);
  EXPECT_EQ(text.text, u8"A中");
  EXPECT_EQ(text.bytes, 4);
  EXPECT_EQ(detail::FitRow(u8"中", 1).bytes, 0);
  EXPECT_EQ(detail::FitRow(u8"e\u0301x", 1).bytes, 3);
  EXPECT_EQ(detail::FitRow("a\n\xff", 3).text, "a  ");
  EXPECT_EQ(detail::FitRow("a\n\xff", 3).bytes, 3);
  EXPECT_EQ(detail::FitRow("abc", 0).bytes, 0);
}

TEST(TuiUtil, QueuesFragmentedKeyboardAndMouseInOrder) {
  detail::InputParser parser;
  std::deque<InputEvent> events;
  const auto now = detail::InputParser::Clock::now();
  parser.Feed("\x1b[1;", now);
  parser.Parse(events, now);
  EXPECT_TRUE(events.empty());
  parser.Feed("5A\x1b[<35;7;9M\x1b[<0;7;9M\x1b[<0;7;9m\x1b[<65;7;9M\r", now);
  parser.Parse(events, now);
  ASSERT_EQ(events.size(), 6);
  const auto key = std::get<KeyEvent>(events[0]);
  EXPECT_EQ(key.key, Key::Up);
  EXPECT_TRUE(key.modifiers.control);
  const auto mouse = std::get<MouseEvent>(events[1]);
  EXPECT_EQ(mouse.action, MouseAction::Move);
  EXPECT_EQ(mouse.button, MouseButton::None);
  EXPECT_EQ(mouse.column, 6);
  EXPECT_EQ(mouse.row, 8);
  EXPECT_EQ(std::get<MouseEvent>(events[2]).action, MouseAction::Press);
  EXPECT_EQ(std::get<MouseEvent>(events[3]).action, MouseAction::Release);
  EXPECT_EQ(std::get<MouseEvent>(events[4]).scroll_y, -1);
  EXPECT_EQ(std::get<KeyEvent>(events[5]).key, Key::Enter);
}

TEST(TuiUtil, UnicodeAltControlAndEscapeTimeout) {
  detail::InputParser parser;
  std::deque<InputEvent> events;
  const auto now = detail::InputParser::Clock::now();
  parser.Feed("\xe4\xb8", now);
  parser.Parse(events, now);
  EXPECT_TRUE(events.empty());
  parser.Feed("\xad\x1bx\x03\x1b", now);
  parser.Parse(events, now);
  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(std::get<KeyEvent>(events[0]).character, U'中');
  EXPECT_TRUE(std::get<KeyEvent>(events[1]).modifiers.alt);
  EXPECT_EQ(std::get<KeyEvent>(events[2]).character, U'c');
  EXPECT_TRUE(std::get<KeyEvent>(events[2]).modifiers.control);
  parser.Parse(events, now + std::chrono::milliseconds(81));
  ASSERT_EQ(events.size(), 4);
  EXPECT_EQ(std::get<KeyEvent>(events[3]).key, Key::Escape);
}

TEST(TuiUtil, RejectsMalformedMouseWithoutLosingFollowingKeys) {
  detail::InputParser parser;
  std::deque<InputEvent> events;
  parser.Feed("\x1b[<0;0;1M\x1b[<99999999999999;2;3M\x1b[B");
  parser.Parse(events);
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(std::get<KeyEvent>(events[0]).key, Key::Down);
  EXPECT_EQ(KeyFromVirtualKey(0x26), Key::Up);
  EXPECT_EQ(MapTerminalKey('~', 24), Key::F12);
}
}  // namespace snowberg::tui

namespace snowberg::tui {
TEST(TuiUtil, ModifiedEnterAndBracketedPaste) {
  detail::InputParser parser;
  std::deque<InputEvent> events;
  parser.Feed("\x1b[13;2u\x1b[27;2;13~\x1b[200~a\r\nb\x1b[20");
  parser.Parse(events);
  ASSERT_EQ(events.size(), 2);
  for (const auto &event : events) {
    EXPECT_EQ(std::get<KeyEvent>(event).key, Key::Enter);
    EXPECT_TRUE(std::get<KeyEvent>(event).modifiers.shift);
  }
  parser.Feed("1~\r");
  parser.Parse(events);
  ASSERT_EQ(events.size(), 4);
  EXPECT_EQ(std::get<PasteEvent>(events[2]).text, "a\r\nb");
  EXPECT_EQ(std::get<KeyEvent>(events[3]).key, Key::Enter);
}

TEST(TuiUtil, UnicodeConversionRoundTripsAndReplacesInvalidBytes) {
  EXPECT_EQ(detail::DecodeUTF8(u8"中😀\n"), U"中😀\n");
  EXPECT_EQ(detail::EncodeUTF8(U'😀'), u8"😀");
  EXPECT_EQ(detail::DecodeUTF8("\xff"), U"\ufffd");
  EXPECT_EQ(detail::EncodeUTF8(0xd800), u8"\ufffd");
}
}  // namespace snowberg::tui
