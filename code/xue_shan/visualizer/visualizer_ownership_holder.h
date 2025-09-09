#pragma once
#include "xue_shan/visualizer/visualizer_util.h"

namespace XS::visualizer {

class OwnershipHolder {
 public:
  void Clear();

  void AddFilm(std::shared_ptr<Film> film);
  void AddCamera(std::shared_ptr<Camera> camera);
  void AddMesh(std::shared_ptr<Mesh> mesh);
  void AddEntity(std::shared_ptr<Entity> entity);

 private:
  std::vector<std::shared_ptr<Film>> held_films_;
  std::vector<std::shared_ptr<Camera>> held_cameras_;
  std::vector<std::shared_ptr<Mesh>> held_meshes_;
  std::vector<std::shared_ptr<Entity>> held_entities_;
};

}  // namespace XS::visualizer
