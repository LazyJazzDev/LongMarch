#pragma once
#include "snow_mount/visualizer/visualizer_util.h"

namespace snow_mount::visualizer {
class Program {
 public:
  virtual ~Program() = default;
};

class ProgramNoNormal : public Program {
 public:
  ~ProgramNoNormal() override = default;

  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> geometry_shader_;
  std::unique_ptr<graphics::Shader> fragment_shader_;
  std::unique_ptr<graphics::Program> program_;
};

}  // namespace snow_mount::visualizer
