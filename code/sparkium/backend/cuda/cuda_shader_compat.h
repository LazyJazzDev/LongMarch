#pragma once
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sparkium::backend::cuda {
// Compatibility adapter for older raw-HLSL shader inputs. The explicit
// compute contract (including all Sparkium shaders) never calls these functions.
std::string LowerLegacySource(std::string source, const std::string &filename, bool optix = false);
std::string RenameLegacyEntry(std::string source, const std::string &entry, const std::string &replacement);
}  // namespace sparkium::backend::cuda
