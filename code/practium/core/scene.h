#pragma once
#include "practium/core/core_util.h"

namespace practium {

class Scene {
 public:
  Scene(Core *core);

 private:
  Core *core_;
  std::unique_ptr<sparkium::Scene> scene_;
  std::unique_ptr<contradium::PBDSolver> pbd_solver_;
};

}  // namespace practium
