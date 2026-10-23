#include "snowberg/tui/tui.h"

#if defined(_WIN32)
#include <windows.h>
#endif

int main(int argc, char **argv) {
#if defined(_WIN32)
  DWORD input_mode{}, output_mode{};
  const auto input = GetStdHandle(STD_INPUT_HANDLE);
  const auto output = GetStdHandle(STD_OUTPUT_HANDLE);
  GetConsoleMode(input, &input_mode);
  GetConsoleMode(output, &output_mode);
  const auto code_page = GetConsoleOutputCP();
  const auto input_code_page = GetConsoleCP();
  CONSOLE_CURSOR_INFO cursor{};
  GetConsoleCursorInfo(output, &cursor);
  cursor.bVisible = FALSE;
  SetConsoleCursorInfo(output, &cursor);
#endif
  std::vector<snowberg::tui::MenuOption> options;
  for (int i = 0; i < 20; ++i)
    options.push_back({"item" + std::to_string(i), "A long description that eventually reveals END_OF_DESCRIPTION"});
  snowberg::tui::Menu menu;
  const auto result = menu.Select("TUI integration probe", options);
  if (argc > 1 && std::string(argv[1]) == "--reuse") {
    const auto next = menu.Select("Second selection", {{"again", "Cancel this selection"}});
    if (next)
      return 101;
  }
#if defined(_WIN32)
  DWORD new_input{}, new_output{};
  GetConsoleMode(input, &new_input);
  GetConsoleMode(output, &new_output);
  CONSOLE_CURSOR_INFO restored_cursor{};
  GetConsoleCursorInfo(output, &restored_cursor);
  if (new_input != input_mode || new_output != output_mode || GetConsoleOutputCP() != code_page ||
      GetConsoleCP() != input_code_page || restored_cursor.bVisible != cursor.bVisible ||
      restored_cursor.dwSize != cursor.dwSize)
    return 100;
#endif
  return result ? static_cast<int>(*result) + 10 : 0;
}
