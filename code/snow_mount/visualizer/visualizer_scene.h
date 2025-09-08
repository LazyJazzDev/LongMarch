#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace XS::visualizer {

class Scene {
  friend class Core;
  Scene(const std::shared_ptr<Core> &core);

 public:
  std::shared_ptr<Core> GetCore() const;

  uint64_t AddEntity(const std::shared_ptr<Entity> &entity);

  static void PyBind(pybind11::module_ &m);

 private:
  std::shared_ptr<Core> core_;
  std::map<uint64_t, std::weak_ptr<Entity>> entities_;
  uint64_t entity_next_id_{0};
};

}  // namespace XS::visualizer
