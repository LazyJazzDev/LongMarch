#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sparkium/core/camera.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"

namespace sparkium {

// Owns every GPU resource referenced by a scene loaded from JSON. Asset paths in
// the document are resolved against the directory containing the JSON file.
class JsonScene {
 public:
  static std::unique_ptr<JsonScene> Load(Core *core, const std::filesystem::path &path, std::string *error = nullptr);

  Scene *GetScene() const {
    return scene_.get();
  }

  Camera *GetCamera() const {
    return camera_.get();
  }

  Film *GetFilm() const {
    return film_.get();
  }

  RenderPipeline GetRenderPipeline() const {
    return render_pipeline_;
  }

  const std::string &GetName() const {
    return name_;
  }

  const std::filesystem::path &GetPath() const {
    return path_;
  }

 private:
  Core *core_{};
  std::filesystem::path path_;
  std::string name_;
  RenderPipeline render_pipeline_{RENDER_PIPELINE_AUTO};
  std::unique_ptr<Scene> scene_;
  std::unique_ptr<Film> film_;
  std::unique_ptr<Camera> camera_;
  std::map<std::string, std::unique_ptr<Material>> materials_;
  std::map<std::string, std::unique_ptr<Geometry>> geometries_;
  std::vector<std::unique_ptr<graphics::Image>> images_;
  std::vector<std::unique_ptr<Entity>> entities_;
};

std::vector<std::filesystem::path> FindJsonScenes(const std::filesystem::path &directory);

}  // namespace sparkium
