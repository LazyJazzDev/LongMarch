#pragma once
#include <optional>
#include <string>
#include <vector>

#include "snowberg/tui/tui_core.h"

namespace snowberg::tui {

struct MenuOption {
  std::string name;
  std::string description;
};

// Blocks until Enter/double-click confirms, or Esc/Ctrl+C/EOF cancels.
// Owns terminal input/output while active; do not run menus or console writers concurrently.
// Redirected streams and terminals without usable capabilities/dimensions use a
// sequential number/name prompt instead of terminal control sequences.
// Returns the zero-based option index, or nullopt on cancellation/terminal failure/empty options.
std::optional<size_t> Select(const std::string &title,
                             const std::vector<MenuOption> &options,
                             size_t initial_selection = 0);

}  // namespace snowberg::tui
