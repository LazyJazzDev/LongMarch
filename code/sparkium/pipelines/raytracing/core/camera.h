#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Camera : public Object {
 public:
  rhi::IShader *Shader();
  virtual rhi::IBuffer *Buffer() = 0;
  const std::string &ShaderFile() const;
  const std::string &EntryPoint() const;

 protected:
  Camera(Core *core, const std::string &shader_file, const std::string &entry_point);

 private:
  Core *core_;
  std::string shader_file_;
  std::string entry_point_;
  rhi::ObjPtr<rhi::IShader> camera_shader_;
};

Camera *DedicatedCast(sparkium::Camera *camera);

}  // namespace sparkium::raytracing
