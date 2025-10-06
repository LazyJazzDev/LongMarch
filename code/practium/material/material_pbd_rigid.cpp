#include "practium/material/material_pbd_rigid.h"

namespace practium {

MaterialPBDRigid::MaterialPBDRigid(Core *core, float mass, float inertia, bool fixed) : Material(core) {
  this->mass = mass;
  this->inertia = inertia;
  if (fixed) {
    this->mass = 0.0f;
    this->inertia = 0.0f;
  }
}

}  // namespace practium
