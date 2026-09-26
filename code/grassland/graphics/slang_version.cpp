#include "grassland/graphics/slang_version.h"

#include <array>
#include <regex>

namespace grassland::graphics::detail {
namespace {
bool ParseVersion(const std::string &tag, std::array<unsigned long, 4> &version) {
  // Official releases and git-describe builds based on a release. Unknown or
  // prerelease tags fail closed rather than silently accepting an old compiler.
  static const std::regex pattern(R"(^v?([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?(?:-[0-9]+-g[0-9a-fA-F]+)?$)");
  std::smatch match;
  if (!std::regex_match(tag, match, pattern))
    return false;
  try {
    version = {std::stoul(match[1]), std::stoul(match[2]), match[3].matched ? std::stoul(match[3]) : 0ul,
               match[4].matched ? std::stoul(match[4]) : 0ul};
  } catch (const std::exception &) {
    return false;
  }
  return true;
}
}  // namespace

const char *MinimumSlangVersion() {
  return LONGMARCH_MIN_SLANG_VERSION;
}

bool SlangVersionSupported(const std::string &build_tag) {
  std::array<unsigned long, 4> actual{}, minimum{};
  return ParseVersion(build_tag, actual) && ParseVersion(MinimumSlangVersion(), minimum) && actual >= minimum;
}
}  // namespace grassland::graphics::detail
