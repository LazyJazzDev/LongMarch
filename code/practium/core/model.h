#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Model {
 public:
  Model(Core *core);
  virtual ~Model() = default;
  virtual Mesh<> VisualMesh() = 0;
  virtual Mesh<> CollisionMesh() = 0;
  virtual sparkium::Material *VisualMaterial() = 0;

 protected:
  Core *core_;
};

}  // namespace practium
