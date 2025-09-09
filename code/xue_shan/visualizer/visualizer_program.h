#pragma once
#include "xue_shan/visualizer/visualizer_util.h"

namespace XS::visualizer {
class Program {
 public:
  virtual ~Program() = default;
};

class ProgramCommonRaster : public Program {
 public:
  ~ProgramCommonRaster() override = default;

  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> fragment_shader_;
  std::unique_ptr<graphics::Program> program_;
};

class ProgramWithGeometryShader : public Program {
 public:
  ~ProgramWithGeometryShader() override = default;

  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> geometry_shader_;
  std::unique_ptr<graphics::Shader> fragment_shader_;
  std::unique_ptr<graphics::Program> program_;
};

}  // namespace XS::visualizer
