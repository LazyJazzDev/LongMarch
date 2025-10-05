#pragma once
#include "practium/material/material_util.h"

namespace practium {

class MaterialPBDRigid : public Material {
 public:
  MaterialPBDRigid(Core *core, float mass = 1.0f, float inertia = 1.0f, bool fixed = false);
  float mass;
  float inertia;
};

}  // namespace practium
