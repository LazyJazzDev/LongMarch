#pragma once

#include <filesystem>

#include "sparkium/scene/scene_definition.h"

namespace sparkium {
// Resolves all external resources eagerly. The result has no file dependencies.
// Throws on invalid input; neither device creation nor shader compilation occurs.
std::shared_ptr<const SceneDefinition> LoadScene(const std::filesystem::path &path);
std::vector<std::filesystem::path> FindJsonScenes(const std::filesystem::path &directory);
}  // namespace sparkium
