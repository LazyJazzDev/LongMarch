#pragma once
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sparkium::backend {
// Compatibility adapter for older raw-HLSL shader inputs. The explicit
// native contract (including all Sparkium shaders) never calls these functions.
std::string LowerLegacySource(std::string source, const std::string &filename, bool optix = false);
std::string RenameLegacyEntry(std::string source, const std::string &entry, const std::string &replacement);
std::set<std::string> LegacyConstantBlocks(const std::string &source);
std::string LowerLegacyHostSource(std::string source, const std::vector<std::pair<std::string, std::string>> &names);
}  // namespace sparkium::backend
