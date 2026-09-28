#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Camera : public Object {
 public:
  graphics::Shader *Shader();
  virtual graphics::Buffer *Buffer() = 0;
  const std::string &ShaderFile() const;
  const std::string &EntryPoint() const;

 protected:
  Camera(Core *core, const std::string &shader_file, const std::string &entry_point);

 private:
  Core *core_;
  std::string shader_file_;
  std::string entry_point_;
  std::unique_ptr<graphics::Shader> camera_shader_;
};

Camera *DedicatedCast(sparkium::Camera *camera);

}  // namespace sparkium::raytracing
