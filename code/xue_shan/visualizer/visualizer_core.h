#pragma once
#include "xue_shan/visualizer/visualizer_scene.h"
#include "xue_shan/visualizer/visualizer_util.h"

namespace XS::visualizer {

class Core : public std::enable_shared_from_this<Core> {
  Core(graphics::Core *core);

 public:
  template <class ProgramType>
  std::shared_ptr<ProgramType> LoadProgram(uint64_t id, const std::function<std::shared_ptr<ProgramType>()> &loader) {
    auto it = programs_.find(id);
    if (it != programs_.end()) {
      return std::static_pointer_cast<ProgramType>(it->second);
    }
    return std::static_pointer_cast<ProgramType>(programs_[id] = loader());
  }

  graphics::Core *GraphicsCore() const;

  static std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core);

  std::shared_ptr<Camera> CreateCamera(const Matrix4<float> &proj, const Matrix4<float> &view);

  std::shared_ptr<Mesh> CreateMesh();

  std::shared_ptr<Film> CreateFilm(int width, int height);

  std::shared_ptr<Scene> CreateScene();

  template <class EntityType, class... Args>
  std::shared_ptr<EntityType> CreateEntity(Args &&...args) {
    return std::make_shared<EntityType>(shared_from_this(), std::forward<Args>(args)...);
  }

  int Render(graphics::CommandContext *context,
             const std::shared_ptr<Scene> &scene,
             const std::shared_ptr<Camera> &camera,
             const std::shared_ptr<Film> &film);

  static void PyBind(pybind11::module_ &m);

 private:
  graphics::Core *core_;
  std::map<uint64_t, std::shared_ptr<Program>> programs_;
  std::vector<OwnershipHolder> ownership_holders_;
};

std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core);

}  // namespace XS::visualizer
