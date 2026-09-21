#pragma once

#include <filesystem>

#include "sparkium/renderer/render_types.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium {
// Resolves all external resources eagerly. The result has no file dependencies.
// Throws on invalid input; neither device creation nor shader compilation occurs.
struct SceneDocument {
  std::shared_ptr<const SceneDefinition> scene;
  RenderPipeline preferred_pipeline{RENDER_PIPELINE_AUTO};
};

SceneDocument LoadSceneDocument(const std::filesystem::path &path);
std::shared_ptr<const SceneDefinition> LoadScene(const std::filesystem::path &path);
std::vector<std::filesystem::path> FindJsonScenes(const std::filesystem::path &directory);
}  // namespace sparkium
