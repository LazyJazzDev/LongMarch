# Terminal selection menu

Link `snowberg_tui` and include `snowberg/tui/tui.h`. The module has no graphics
dependencies. `Select(title, options, initial_selection)` returns a zero-based
index or `std::nullopt` on cancellation, EOF, an empty menu or terminal failure.

```cpp
const auto choice = snowberg::tui::Select("Example", {
    {"triangle", "Colored triangle"},
    {"texture", "Sampled texture"},
});
if (!choice)
  return;
```

- Up/Down moves the highlight; Enter confirms; Esc or Ctrl+C cancels.
- Left-click selects a row, double-click confirms; the mouse wheel moves selection.
- Every option occupies one row. Width changes crop the right edge without wrapping.
  The selected row pauses for one second, scrolls left at 12.5 cells/second,
  pauses at the end, and repeats. Changing selection or resizing restarts scrolling.
- Short terminals scroll vertically to keep the selected row visible. Title and
  help rows are hidden when fewer than four rows are available.
- UTF-8 text supports CJK full-width characters and common combining accents;
  control characters are replaced with spaces. Complex emoji clusters and
  terminal-specific ambiguous-width characters are not normalized.
- Windows uses console input events; Linux/macOS use termios and xterm-compatible
  arrow/mouse sequences. Terminal input modes, mouse reporting, cursor visibility
  and the alternate screen are cleaned up when selection ends. Don't run other
  console writers or menus concurrently. Forced process termination cannot restore
  terminal state.
- Redirected streams and unsupported terminals use a number/name prompt. Invalid
  entries retry; `q`, `quit`, Esc or EOF cancels. CLI demo selection remains available
  through `--module` without entering terminal mode.

Tests: `test_tui` exercises UTF-8 clipping and animation boundaries;
`test/tui/test_console.py` drives native Windows console keyboard/mouse/resize events
against `tui_menu_probe` and the three demo launchers.
