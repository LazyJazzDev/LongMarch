#pragma once

#include <string>

namespace grassland::graphics::detail {
// Check the loaded compiler's build tag, not the SDK headers' version.
bool SlangVersionSupported(const std::string &build_tag);
const char *MinimumSlangVersion();
}  // namespace grassland::graphics::detail
