#include "snow_mount/visualizer/visualizer_ownership_holder.h"

namespace XS::visualizer {

void OwnershipHolder::Clear() {
  held_films_.clear();
  held_cameras_.clear();
  held_meshes_.clear();
  held_entities_.clear();
}

void OwnershipHolder::AddFilm(std::shared_ptr<Film> film) {
  held_films_.emplace_back(std::move(film));
}

void OwnershipHolder::AddCamera(std::shared_ptr<Camera> camera) {
  held_cameras_.emplace_back(std::move(camera));
}

void OwnershipHolder::AddMesh(std::shared_ptr<Mesh> mesh) {
  held_meshes_.emplace_back(std::move(mesh));
}

void OwnershipHolder::AddEntity(std::shared_ptr<Entity> entity) {
  held_entities_.emplace_back(std::move(entity));
}

}  // namespace XS::visualizer
